SELECT  -- DISTINCT o.id,
        -- o.name,
        o.category_code,
        COUNT(o.category_code)
        -- f.funding_round_type,
        -- f.funding_round_code,
        -- f.funded_at,
        -- f.raised_amount_usd,
        -- TIMESTAMPDIFF(DAY, o.founded_at, f.funded_at) AS time_to_funding
FROM cb_objects AS o 
JOIN cb_funding_rounds AS f 
ON o.id = f.object_id
-- WHERE o.name = 'twitter'
-- ORDER BY f.funded_at DESC
WHERE f.raised_amount_usd IS NOT NULL
    AND TIMESTAMPDIFF(DAY, o.founded_at, f.funded_at) IS NOT NULL
GROUP BY o.category_code
ORDER BY COUNT(o.category_code) DESC
-- LIMIT 50

-- mysql> source funding_rounds_query.sql
-- +------------------+------------------------+
-- | category_code    | COUNT(o.category_code) |
-- +------------------+------------------------+
-- | software         |                   5900 |
-- | biotech          |                   4008 |
-- | web              |                   3126 |
-- | mobile           |                   2790 |
-- | enterprise       |                   2676 |
-- | advertising      |                   1888 |
-- | ecommerce        |                   1842 |
-- | games_video      |                   1562 |
-- | cleantech        |                   1320 |
-- | hardware         |                   1279 |
-- | analytics        |                   1114 |
-- | medical          |                    976 |
-- | social           |                    777 |
-- | semiconductor    |                    771 |
-- | finance          |                    661 |
-- | health           |                    652 |
-- | network_hosting  |                    644 |
-- | education        |                    611 |
-- | security         |                    584 |
-- | other            |                    531 |
-- | search           |                    418 |
-- | public_relations |                    355 |
-- | manufacturing    |                    338 |
-- | consulting       |                    313 |
-- | messaging        |                    310 |
-- | travel           |                    282 |
-- | fashion          |                    271 |
-- | news             |                    270 |
-- | hospitality      |                    262 |
-- | music            |                    256 |
-- | photo_video      |                    182 |
-- | real_estate      |                    165 |
-- | sports           |                    140 |
-- | automotive       |                    132 |
-- | nanotech         |                    120 |
-- | transportation   |                    119 |
-- | nonprofit        |                    115 |
-- | legal            |                     74 |
-- | design           |                     49 |
-- | local            |                     30 |
-- | pets             |                     24 |
-- | government       |                      7 |
-- | NULL             |                      0 |
-- +------------------+------------------------+
-- 43 rows in set (0.24 sec)
