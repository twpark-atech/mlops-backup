# src/pipelines/bronze_to_silver.py

import argparse
from datetime import datetime, timedelta

from pyspark.sql import functions as F

from src.common.spark_session import create_spark

# 🔹 Ulsan 타겟 링크들 (예전 Dask 코드에서 쓰던 리스트 그대로)
LINKIDS_ALL = [
    "1920161400", "1920161500",
    "1920121301", "1920121401",
    "1920161902", "1920162205", "1920162400",
    "1920000702", "1920000801", "1920121000", "1920121302", "1920121402",
    "1920235801", "1920189001", "1920139400", "1920161801", "1920162207",
    "1920162304", "1920162500", "1920171200", "1920171600", "1920188900", "1920138500",
]


def run_its_traffic_bronze_to_silver(start_date: str, end_date: str) -> None:
    """
    BRONZE (raw 정리된 parquet) -> SILVER (5분·링크 단위 평균 속도)
    - 입력:  s3a://its/traffic/bronze/date=YYYYMMDD/*.parquet
    - 출력:  s3a://its/traffic/silver/date=YYYYMMDD
    """

    print(f"[BRONZE→SILVER] job=its_traffic_5min, range={start_date} ~ {end_date}")

    spark = create_spark("BRONZE_TO_SILVER_ITS_TRAFFIC_5MIN")

    base_bronze = "s3a://its/traffic/bronze"
    base_silver = "s3a://its/traffic/silver"

    start = datetime.strptime(start_date, "%Y%m%d")
    end = datetime.strptime(end_date, "%Y%m%d")

    cur = start
    while cur <= end:
        dt = cur.strftime("%Y%m%d")

        input_path = f"{base_bronze}/date={dt}"
        output_path = f"{base_silver}/date={dt}"

        print(f"[BRONZE→SILVER] {dt} 읽는 중: {input_path}")

        try:
            df_bronze = spark.read.parquet(input_path)

            # 🔹 필요한 컬럼만 가져오고 캐스팅
            df = (
                df_bronze.select(
                    F.col("CREATDE").cast("string"),
                    F.col("CREATHM").cast("string"),
                    F.col("LINKID").cast("string"),
                    F.col("PASNGSPED").cast("double"),
                )
                # 날짜 / 시각 정규화
                .withColumn("CREATDE", F.lpad(F.col("CREATDE"), 8, "0"))
                .withColumn("CREATHM", F.lpad(F.col("CREATHM"), 4, "0"))
            )

            # 🔹 Ulsan 대상 링크만 필터 (메모리 줄이기)
            df = df.filter(F.col("LINKID").isin(LINKIDS_ALL))

            # 🔹 유효한 날짜·시간만 남기기
            df = df.filter(
                (F.col("CREATDE").rlike(r"^\d{8}$")) &
                (F.col("CREATHM").rlike(r"^\d{4}$"))
            )

            # 🔹 datetime 생성 (분 단위)
            df = df.withColumn(
                "datetime",
                F.to_timestamp(F.concat(F.col("CREATDE"), F.col("CREATHM")), "yyyyMMddHHmm")
            )

            # 🔹 5분 버킷으로 내림 (2025-11-13 10:03 -> 10:00, 10:07 -> 10:05)
            df = df.withColumn(
                "minute",
                F.minute("datetime")
            ).withColumn(
                "minute_5",
                (F.col("minute") / 5).cast("int") * 5
            ).withColumn(
                "datetime_5min",
                F.concat_ws(
                    " ",
                    F.date_format("datetime", "yyyy-MM-dd"),
                    F.format_string(
                        "%02d:%02d:00",
                        F.hour("datetime"),
                        F.col("minute_5")
                    )
                ).cast("timestamp")
            )

            # 🔹 당일 데이터만 (CREATDE = dt)
            df = df.filter(F.col("CREATDE") == dt)

            # 🔹 5분·링크 단위 평균 속도 (= self_mean)
            df_silver = (
                df.groupBy("datetime_5min", "LINKID")
                  .agg(F.avg("PASNGSPED").alias("self_mean"))
                  .withColumnRenamed("LINKID", "linkid")
                  .withColumnRenamed("datetime_5min", "datetime")
            )

            # 여기서 .count() 하지 말자 → 바로 write (OOM 방지)
            # row_cnt = df_silver.count()
            # print(f"[BRONZE→SILVER] {dt} row 수: {row_cnt}")

            # 🔹 SILVER 저장 (5분·링크 평균 속도)
            (
                df_silver
                .repartition(8, "datetime")  # 시간 기준 파티셔닝 (적당히 나눔)
                .write.mode("overwrite")
                .parquet(output_path)
            )

            print(f"[BRONZE→SILVER] {dt} → {output_path} 저장 완료")

        except Exception as e:
            print(f"[BRONZE→SILVER][WARN] {dt} 처리 실패: {e}")

        cur += timedelta(days=1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-name", type=str, required=True)
    parser.add_argument("--start-date", type=str, required=True)  # YYYYMMDD
    parser.add_argument("--end-date", type=str, required=True)    # YYYYMMDD
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_its_traffic_bronze_to_silver(args.start_date, args.end_date)
