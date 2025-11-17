# src/pipelines/raw_to_bronze.py

from datetime import datetime, timedelta
import argparse

from src.common.spark_session import create_spark


def run_its_traffic_raw_to_bronze(start_date: str, end_date: str) -> None:
    """
    ITS 5분 교통 CSV (raw) → bronze(parquet) 변환
    - 입력:  s3a://its/traffic/raw/date=YYYYMMDD/*.csv
    - 출력: s3a://its/traffic/bronze/date=YYYYMMDD/
    """
    spark = create_spark("RAW_TO_BRONZE_ITS_TRAFFIC_5MIN")

    # MinIO 경로
    base_raw = "s3a://its/traffic/raw"
    base_bronze = "s3a://its/traffic/bronze"

    start = datetime.strptime(start_date, "%Y%m%d")
    end = datetime.strptime(end_date, "%Y%m%d")

    from pyspark.sql import functions as F

    cur = start
    while cur <= end:
        dt = cur.strftime("%Y%m%d")
        input_path = f"{base_raw}/date={dt}/*.csv"
        output_path = f"{base_bronze}/date={dt}"

        print(f"[RAW→BRONZE] {dt} 읽는 중: {input_path}")

        try:
            df_raw = (
                spark.read
                .option("header", True)       # 헤더 있으면 True, 없으면 False
                .option("inferSchema", True)  # 일단 스키마 추론
                .csv(input_path)
            )

            # 컬럼/타입 정리
            df_clean = (
                df_raw
                .select(
                    F.col("CREATDE").cast("string"),
                    F.col("CREATHM").cast("string"),
                    F.col("LINKID").cast("string"),
                    F.col("ROADINSTTCD").cast("string"),
                    F.col("PASNGSPED").cast("double"),
                    F.col("PASNGTIME").cast("double"),
                )
            )

            cnt = df_clean.count()
            print(f"[RAW→BRONZE] {dt} row 수: {cnt}")

            (
                df_clean
                .repartition(1)
                .write.mode("overwrite")
                .parquet(output_path)
            )
            print(f"[RAW→BRONZE] {dt} → {output_path} 저장 완료")

        except Exception as e:
            print(f"[RAW→BRONZE][WARN] {dt} 처리 실패: {e}")

        cur += timedelta(days=1)


def main():
    parser = argparse.ArgumentParser(description="ITS traffic 5min RAW → BRONZE 변환")
    parser.add_argument(
        "--job-name",
        type=str,
        default="its_traffic_5min",
        help="논리적 잡 이름 (로깅용)",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        required=True,
        help="시작 일자 (YYYYMMDD)",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        required=True,
        help="종료 일자 (YYYYMMDD)",
    )

    args = parser.parse_args()

    print(
        f"[RAW→BRONZE] job={args.job_name}, "
        f"range={args.start_date} ~ {args.end_date}"
    )
    run_its_traffic_raw_to_bronze(args.start_date, args.end_date)


if __name__ == "__main__":
    main()
