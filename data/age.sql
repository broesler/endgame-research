SELECT b.name AS acquired,
       MIN(a.acquired_at)
FROM cb_objects AS b 
JOIN cb_acquisitions AS a 
ON b.id = a.acquired_object_id 
GROUP BY b.id;
