-- Small analytics schema. Deliberately has near-duplicate column names and one
-- denormalised table, so schema discovery is a real step and not a formality.

CREATE TABLE customers (
    customer_id   SERIAL PRIMARY KEY,
    full_name     TEXT NOT NULL,
    country_code  CHAR(2) NOT NULL,
    signup_date   DATE NOT NULL,
    is_active     BOOLEAN NOT NULL DEFAULT TRUE
);

CREATE TABLE products (
    product_id    SERIAL PRIMARY KEY,
    title         TEXT NOT NULL,
    category      TEXT NOT NULL,
    unit_price    NUMERIC(10,2) NOT NULL,
    discontinued  BOOLEAN NOT NULL DEFAULT FALSE
);

CREATE TABLE orders (
    order_id      SERIAL PRIMARY KEY,
    customer_id   INT NOT NULL REFERENCES customers(customer_id),
    order_date    DATE NOT NULL,
    status        TEXT NOT NULL CHECK (status IN ('placed','shipped','delivered','cancelled','refunded'))
);

CREATE TABLE order_items (
    order_item_id SERIAL PRIMARY KEY,
    order_id      INT NOT NULL REFERENCES orders(order_id),
    product_id    INT NOT NULL REFERENCES products(product_id),
    quantity      INT NOT NULL,
    unit_price    NUMERIC(10,2) NOT NULL   -- price AT TIME OF ORDER, not products.unit_price
);

-- Denormalised support table: overlaps orders, tempts the model into the wrong join.
CREATE TABLE support_tickets (
    ticket_id     SERIAL PRIMARY KEY,
    order_id      INT REFERENCES orders(order_id),
    customer_name TEXT NOT NULL,           -- duplicated from customers.full_name, can be stale
    opened_at     TIMESTAMP NOT NULL,
    resolved_at   TIMESTAMP,
    severity      TEXT NOT NULL CHECK (severity IN ('low','medium','high'))
);

INSERT INTO customers (full_name, country_code, signup_date, is_active)
SELECT 'Customer ' || i,
       (ARRAY['ES','FR','DE','IT','GB','US'])[1 + (i % 6)],
       DATE '2024-01-01' + (i % 500),
       (i % 9) <> 0
FROM generate_series(1, 400) AS i;

INSERT INTO products (title, category, unit_price, discontinued)
SELECT 'Product ' || i,
       (ARRAY['audio','video','storage','network','power'])[1 + (i % 5)],
       ROUND((5 + (i * 7 % 300))::numeric, 2),
       (i % 17) = 0
FROM generate_series(1, 120) AS i;

INSERT INTO orders (customer_id, order_date, status)
SELECT 1 + (i * 13 % 400),
       DATE '2025-01-01' + (i % 400),
       (ARRAY['placed','shipped','delivered','delivered','cancelled','refunded'])[1 + (i % 6)]
FROM generate_series(1, 3000) AS i;

INSERT INTO order_items (order_id, product_id, quantity, unit_price)
SELECT 1 + (i % 3000),
       1 + (i * 29 % 120),
       1 + (i % 5),
       ROUND((5 + (i * 11 % 280))::numeric, 2)
FROM generate_series(1, 9000) AS i;

INSERT INTO support_tickets (order_id, customer_name, opened_at, severity, resolved_at)
SELECT 1 + (i * 7 % 3000),
       'Customer ' || (1 + (i * 3 % 400)),
       TIMESTAMP '2025-02-01 00:00:00' + (i % 300) * INTERVAL '1 day',
       (ARRAY['low','medium','high'])[1 + (i % 3)],
       CASE WHEN i % 4 = 0 THEN NULL
            ELSE TIMESTAMP '2025-02-01 00:00:00' + (i % 300) * INTERVAL '1 day' + INTERVAL '6 hours' END
FROM generate_series(1, 900) AS i;

-- Read-only role the agent connects as. Cannot mutate, so a hallucinated
-- DELETE fails loudly instead of corrupting the fixture.
CREATE USER agent_ro WITH PASSWORD 'agent_ro';
GRANT CONNECT ON DATABASE shop TO agent_ro;
GRANT USAGE ON SCHEMA public TO agent_ro;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO agent_ro;
