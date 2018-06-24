SELECT o.id,
       o.name,
       -- a1.acquired_at < MIN(a2.acquired_at) AS is_before_acq
       a1.acquired_at,
       a1.term_code,
       a1.price_amount
FROM cb_objects AS o 
JOIN cb_acquisitions AS a1
ON o.id = a1.acquiring_object_id 
-- JOIN cb_acquisitions AS a2
-- ON o.id = a2.acquired_object_id
-- WHERE a1.price_amount IS NOT NULL
-- GROUP BY o.id, a1.acquired_at
