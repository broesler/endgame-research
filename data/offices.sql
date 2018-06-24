SELECT o.name, 
       o.id, 
       l.latitude, 
       l.longitude,
       l.description
FROM cb_objects AS o 
JOIN cb_offices AS l   
ON o.id = l.object_id 
-- WHERE l.description RLIKE 'HQ|headquarters|main' 
WHERE o.name = 'wetpaint'
    AND o.entity_type = 'company' 
LIMIT 50;
