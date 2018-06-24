SELECT f.funding_round_code,
       COUNT(f.funding_round_code)
FROM cb_objects AS o 
JOIN cb_funding_rounds AS f 
ON o.id = f.object_id
GROUP BY f.funding_round_code
ORDER BY COUNT(f.funding_round_code) DESC
-- LIMIT 50
