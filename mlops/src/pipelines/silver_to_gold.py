# src/pipelines/silver_to_gold.py

import os
import argparse
from datetime import datetime, timedelta

from pyspark.sql import functions as F
from src.common.spark_session import create_spark

import psycopg2


def _get_env(name: str, default: str) -> str:
    """환경 변수 읽기 (없으면 default)."""
    return os.environ.get(name, default)


def _delete_existing_partition(
    dt: str,
    table: str,
    host: str,
    port: str,
    db: str,
    user: str,
    password: str,
    date_col: str = "date",
) -> None:
    """
    재실행 시 중복 방지용: GOLD 테이블에서 해당 날짜 파티션을 미리 삭제.
    """
    conn = None
    try:
        conn = psycopg2.connect(
            host=host,
            port=port,
            dbname=db,
            user=user,
            password=password,
        )
        conn.autocommit = True
        with conn.cursor() as cur:
            sql = f'DELETE FROM "{table}" WHERE "{date_col}" = %s'
            cur.execute(sql, (dt,))
        print(f'[SILVER→GOLD] {table} 에서 date={dt} 기존 행 삭제 완료')
    except Exception as e:
        print(f'[SILVER→GOLD][WARN] date={dt} 삭제 중 오류 (무시하고 진행): {e}')
    finally:
        if conn is not None:
            conn.close()


def run_its_traffic_silver_to_gold(start_date: str, end_date: str) -> None:
    """
    ITS 5분 단위 교통 속도 Silver → Postgres GOLD 적재
    """

    # 🔹 Spark 세션 (S3 + JDBC 드라이버가 설정된 상태여야 함)
    spark = create_spark("SILVER_TO_GOLD_ITS_TRAFFIC_5MIN")

    # 🔹 Silver 위치 (MinIO / S3A)
    base_silver = "s3a://its/traffic/silver"

    # 🔹 Postgres 접속 정보 (환경변수 기반)
    pg_host = _get_env("PG_HOST", "localhost")
    pg_port = _get_env("PG_PORT", "5431")
    pg_db = _get_env("PG_DB", "mlops")
    pg_user = _get_env("PG_USER", "postgres")
    pg_password = _get_env("PG_PASSWORD", "postgres")
    pg_table = _get_env("ITS_TRAFFIC_GOLD_TABLE", "its_traffic_5min_gold")

    jdbc_url = f"jdbc:postgresql://{pg_host}:{pg_port}/{pg_db}"

    start = datetime.strptime(start_date, "%Y%m%d")
    end = datetime.strptime(end_date, "%Y%m%d")

    cur = start
    while cur <= end:
        dt = cur.strftime("%Y%m%d")
        silver_path = f"{base_silver}/date={dt}"

        print(f"[SILVER→GOLD] {dt} 읽는 중: {silver_path}")

        try:
            df_silver = spark.read.parquet(silver_path)

            # ✅ 스키마 살짝 정리
            # - datetime: timestamp 그대로 사용
            # - linkid: string
            # - speed_mean / count 같은 숫자 컬럼은 double/long 유지
            cols = df_silver.columns

            # 컬럼 이름 예시가 다를 수 있으니 guard 걸어서 처리
            # (bronze_to_silver에서 어떤 이름 썼는지 맞춰서 필요시 살짝 수정하면 됨)
            rename_map = {}
            if "LINKID" in cols:
                rename_map["LINKID"] = "linkid"
            if "DATETIME_5MIN" in cols:
                rename_map["DATETIME_5MIN"] = "datetime"
            if "datetime_5min" in cols:
                rename_map["datetime_5min"] = "datetime"

            df_out = df_silver
            for src, dst in rename_map.items():
                df_out = df_out.withColumnRenamed(src, dst)

            # date 컬럼은 GOLD 테이블 파티셔닝 / 조회용으로 하나 더 넣어줌
            df_out = df_out.withColumn("date", F.lit(dt).cast("string"))

            # (선택) 컬럼 순서 정리
            ordered_cols = []
            for c in ["date", "datetime", "linkid"]:
                if c in df_out.columns:
                    ordered_cols.append(c)
            # 나머지 메트릭 컬럼들 뒤에 붙이기
            ordered_cols += [c for c in df_out.columns if c not in ordered_cols]
            df_out = df_out.select(*ordered_cols)

            # ✅ 재실행 대비: 해당 날짜 데이터 먼저 삭제
            _delete_existing_partition(
                dt=dt,
                table=pg_table,
                host=pg_host,
                port=pg_port,
                db=pg_db,
                user=pg_user,
                password=pg_password,
                date_col="date",
            )

            # ✅ Postgres에 append
            (
                df_out.write
                .mode("append")
                .format("jdbc")
                .option("url", jdbc_url)
                .option("dbtable", pg_table)
                .option("user", pg_user)
                .option("password", pg_password)
                .option("driver", "org.postgresql.Driver")
                .save()
            )

            print(f"[SILVER→GOLD] {dt} → {pg_table} 적재 완료")

        except Exception as e:
            print(f"[SILVER→GOLD][WARN] {dt} 처리 실패: {e}")

        cur += timedelta(days=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-name", required=True, help="예: its_traffic_5min")
    parser.add_argument("--start-date", required=True, help="YYYYMMDD")
    parser.add_argument("--end-date", required=True, help="YYYYMMDD")
    args = parser.parse_args()

    if args.job_name == "its_traffic_5min":
        run_its_traffic_silver_to_gold(args.start_date, args.end_date)
    else:
        raise ValueError(f"지원하지 않는 job-name: {args.job_name}")


if __name__ == "__main__":
    main()