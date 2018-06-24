SELECT  o.id,
        o.name,
        o.status,
        f.funding_round_type,
        f.funding_round_code,
        f.funded_at,
        f.raised_amount_usd,
        TIMESTAMPDIFF(DAY, o.founded_at, f.funded_at) AS time_to_funding
FROM cb_objects AS o 
JOIN cb_funding_rounds AS f 
ON o.id = f.object_id
-- WHERE f.raised_amount_usd IS NOT NULL
WHERE o.founded_at IS NOT NULL
    -- AND TIMESTAMPDIFF(DAY, o.founded_at, COALESCE(f.funded_at, CURRENT_DATE)) IS NOT NULL
-- LIMIT 60
