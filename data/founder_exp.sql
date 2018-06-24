SELECT o1.name, 
       o2.name AS employee,
       TIMESTAMPDIFF(MONTH, MIN(r2.start_at), '2013/12/12') as exp_months
FROM cb_objects AS o1 
JOIN cb_relationships AS r1 
ON o1.id = r1.relationship_object_id 
JOIN cb_objects AS o2
ON r1.person_object_id = o2.id
JOIN cb_relationships AS r2
ON o2.id = r2.person_object_id
WHERE o1.name = 'wetpaint' 
    AND r1.title RLIKE 'founder|board|director'
GROUP BY o2.name;
