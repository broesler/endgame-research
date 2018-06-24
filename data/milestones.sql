SELECT o.name, 
       m.milestone_at
       -- MIN(a.acquired_at) AS acq_at,
       -- (m.milestone_at < MIN(a.acquired_at)) AS is_before_acq
FROM cb_objects AS o 
JOIN cb_milestones AS m 
ON o.id = m.object_id 
-- JOIN cb_acquisitions AS a 
-- ON o.id = a.acquired_object_id 
-- GROUP BY o.id, m.milestone_at
