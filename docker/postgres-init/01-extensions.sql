-- Runs once, on first initialisation of the data volume.
--
-- The extensions are created here rather than in an Alembic migration because
-- CREATE EXTENSION needs superuser and the application role should not have it.
-- Alembic then assumes pgvector is present and fails loudly if it is not.
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;     -- fuzzy handle / domain search
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Every timestamp in this system is UTC. Setting it on the database means a
-- client that forgets to say so still gets UTC rather than the container's
-- guess at a locale.
DO $$
BEGIN
  EXECUTE format('ALTER DATABASE %I SET timezone TO %L', current_database(), 'UTC');
END
$$;
