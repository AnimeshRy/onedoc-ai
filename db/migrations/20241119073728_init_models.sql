-- migrate:up
CREATE TABLE file_embedding_status (
    id UUID PRIMARY KEY,
    resource_id VARCHAR(255) NOT NULL,
    status VARCHAR(255) NOT NULL,
    document_ids TEXT[],
    root_id CHAR(255),
    meta_info JSONB,
    created_at timestamptz DEFAULT NOW(),
    updated_at timestamptz DEFAULT NOW()
);

-- migrate:down

DROP TABLE file_embedding_status;
