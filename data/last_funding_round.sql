SELECT  o.id,
        o.name,
        o.last_funding_at,
        f.funding_round_code,
        f.funded_at,
        f.raised_amount_usd
FROM cb_objects AS o 
JOIN cb_funding_rounds AS f 
ON o.id = f.object_id
-- WHERE f.raised_amount_usd IS NOT NULL
WHERE o.founded_at IS NOT NULL
    AND o.last_funding_at = f.funded_at
LIMIT 60
