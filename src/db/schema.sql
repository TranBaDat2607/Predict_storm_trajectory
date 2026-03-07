-- Storm trajectory database schema
-- Requires: PostgreSQL + PostGIS extension

CREATE EXTENSION IF NOT EXISTS postgis;

-- ─────────────────────────────────────────────────────────────────────────────
-- nature_types  (lookup)
-- ─────────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS nature_types (
    id          SMALLINT PRIMARY KEY,
    code        CHAR(2)  NOT NULL UNIQUE,
    description TEXT     NOT NULL
);

INSERT INTO nature_types (id, code, description) VALUES
    (0, 'DS', 'Disturbance'),
    (1, 'ET', 'Extratropical'),
    (2, 'MX', 'Mixed/Subtropical'),
    (3, 'SS', 'Subtropical'),
    (4, 'TS', 'Tropical')
ON CONFLICT (id) DO NOTHING;

-- ─────────────────────────────────────────────────────────────────────────────
-- storms  (one row per named storm)
-- ─────────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS storms (
    atcf_id  TEXT     PRIMARY KEY,
    season   SMALLINT NOT NULL CHECK (season BETWEEN 1800 AND 2200),
    basin    CHAR(2)  NOT NULL,
    subbasin CHAR(2)
);

CREATE INDEX IF NOT EXISTS idx_storms_season ON storms (season);
CREATE INDEX IF NOT EXISTS idx_storms_basin  ON storms (basin);

-- ─────────────────────────────────────────────────────────────────────────────
-- storm_observations  (one row per storm × timestep)
-- ─────────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS storm_observations (
    id          BIGSERIAL    PRIMARY KEY,
    atcf_id     TEXT         NOT NULL REFERENCES storms (atcf_id) ON DELETE CASCADE,
    iso_time    TIMESTAMPTZ  NOT NULL,
    lat         REAL         NOT NULL,
    lon         REAL         NOT NULL,
    geom        GEOGRAPHY(POINT, 4326)
                    GENERATED ALWAYS AS (
                        ST_SetSRID(ST_MakePoint(lon, lat), 4326)::GEOGRAPHY
                    ) STORED,
    nature      SMALLINT     NOT NULL REFERENCES nature_types (id),
    dist2land   REAL,
    landfall    REAL,
    wind_speed  REAL,
    storm_pres  REAL,
    usa_sshs    SMALLINT,
    usa_poci    REAL,
    usa_roci    REAL,
    usa_rmw     REAL,
    storm_speed REAL,
    storm_dir   REAL,
    UNIQUE (atcf_id, iso_time)
);

CREATE INDEX IF NOT EXISTS idx_obs_atcf_id   ON storm_observations (atcf_id);
CREATE INDEX IF NOT EXISTS idx_obs_iso_time  ON storm_observations (iso_time);
CREATE INDEX IF NOT EXISTS idx_obs_atcf_time ON storm_observations (atcf_id, iso_time);
CREATE INDEX IF NOT EXISTS idx_obs_geom      ON storm_observations USING GIST (geom);
