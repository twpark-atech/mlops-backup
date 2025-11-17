# airflow/dags/its_traffic_daily.py

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator


default_args = {
    "owner": "mlops",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

# 매일 새벽 4시 (Asia/Seoul 기준) 기준으로 전일 날짜를 처리하고 싶으면
# catchup=True + start_date 설정으로 과거도 돌릴 수 있음.
with DAG(
    dag_id="its_traffic_daily",
    default_args=default_args,
    start_date=datetime(2025, 1, 1),
    schedule_interval="0 4 * * *",   # 매일 04:00
    catchup=True,
    tags=["its", "traffic", "convlstm"],
) as dag:

    # 1) ITS ZIP 수집 + MinIO 업로드 (+ Kafka 이벤트)
    ingest_its = BashOperator(
        task_id="ingest_its",
        bash_command=(
            "cd /opt/mlops && "
            "python -m src.ingestion.file_ingestor.main "
            "--date {{ ds_nodash }}"
        ),
    )

    # 2) bronze → silver 변환 (Spark/Polars 파이프라인)
    bronze_to_silver = BashOperator(
        task_id="bronze_to_silver",
        bash_command=(
            "cd /opt/mlops && "
            "python -m src.pipelines.bronze_to_silver_its "
            "--start-date {{ ds_nodash }} "
            "--end-date {{ ds_nodash }}"
        ),
    )

    # 3) silver → gold (its_traffic_5min_gold 생성/업데이트)
    silver_to_gold = BashOperator(
        task_id="silver_to_gold",
        bash_command=(
            "cd /opt/mlops && "
            "python -m src.pipelines.silver_to_gold_its "
            "--job-name its_traffic_5min "
            "--start-date {{ ds_nodash }} "
            "--end-date {{ ds_nodash }}"
        ),
    )

    # 4) ConvLSTM 학습 (해당 일자 데이터로 incremental 학습 or 재학습)
    train_convlstm = BashOperator(
        task_id="train_convlstm",
        bash_command=(
            "cd /opt/mlops && "
            "python -m src.training.train_its_traffic_convlstm "
            "--job-name its_traffic_5min_convlstm "
            "--start-date {{ ds_nodash }} "
            "--end-date {{ ds_nodash }}"
        ),
    )

    ingest_its >> bronze_to_silver >> silver_to_gold >> train_convlstm
