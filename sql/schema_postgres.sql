-- sql/schema_postgres.sql

-- ITS 5분 단위 골드 피처 테이블
CREATE TABLE IF NOT EXISTS its_traffic_5min_gold (
    date        varchar(8)   NOT NULL,              -- YYYYMMDD
    datetime    timestamp    NOT NULL,              -- 5분 단위 타임스탬프
    linkid      varchar(20)  NOT NULL,              -- 링크 ID

    t2_mean     real         NOT NULL,              -- 2-hop downstream 평균 속도
    t1_mean     real         NOT NULL,              -- 1-hop downstream 평균 속도
    self_mean   real         NOT NULL,              -- 자기 링크 평균 속도
    f1_mean     real         NOT NULL,              -- 1-hop upstream 평균 속도
    f2_mean     real         NOT NULL,              -- 2-hop upstream 평균 속도

    created_at  timestamp    NOT NULL DEFAULT now(),

    CONSTRAINT pk_its_traffic_5min_gold PRIMARY KEY (date, datetime, linkid)
);

CREATE INDEX IF NOT EXISTS idx_its_traffic_5min_gold_link_date
    ON its_traffic_5min_gold (linkid, date);
