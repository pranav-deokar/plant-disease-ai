-- PostgreSQL initialization script
-- Runs once when the postgres container is first created.
-- Creates the MLflow database alongside the main application database.

-- Main application database (already created by POSTGRES_DB env var)
-- Just ensure extensions are enabled:
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";    -- for trigram text search
CREATE EXTENSION IF NOT EXISTS "btree_gin";  -- for composite GIN indexes

-- MLflow database (needed by MLflow tracking server)
CREATE DATABASE mlflow
    WITH OWNER = postgres
    ENCODING = 'UTF8'
    LC_COLLATE = 'en_US.UTF-8'
    LC_CTYPE = 'en_US.UTF-8'
    TEMPLATE = template0;

-- Performance tuning for plant_disease database
ALTER SYSTEM SET shared_buffers = '256MB';
ALTER SYSTEM SET work_mem = '16MB';
ALTER SYSTEM SET maintenance_work_mem = '128MB';
ALTER SYSTEM SET effective_cache_size = '512MB';
ALTER SYSTEM SET max_connections = '100';
ALTER SYSTEM SET wal_level = 'replica';
ALTER SYSTEM SET checkpoint_completion_target = '0.9';
ALTER SYSTEM SET random_page_cost = '1.1';    -- assume SSD

SELECT pg_reload_conf();
