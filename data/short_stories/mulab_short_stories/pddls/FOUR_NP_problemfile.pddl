(define (problem The_Four_Skilful_Brothers_problem)
   (:domain The_Four_Skilful_Brothers)


   (:init 
      (entity_at father house)
      (entity_at brother_thief house)
      (entity_at brother_astronomer house)
      (entity_at brother_hunter house)
      (entity_at brother_tailor house)
      (has brother_thief sticks)
      (has brother_astronomer sticks)
      (has brother_hunter sticks)
      (has brother_tailor sticks)
      (item_at eggs tree)
   )

   (:goal 
      (and (is_thief brother_thief) (is_astronomer brother_astronomer) (is_hunter brother_hunter) (is_tailor brother_tailor) (eggs_fetched brother_thief) (eggs_shot brother_hunter) (eggs_sewn brother_tailor) (item_at eggs tree) (princess_rescued brother_thief) (dragon_dead) (ship_repaired) (reward_received brother_thief) (reward_received brother_astronomer) (reward_received brother_hunter) (reward_received brother_tailor) (half_kingdom brother_thief kingdom) (half_kingdom brother_astronomer kingdom) (half_kingdom brother_hunter kingdom) (half_kingdom brother_tailor kingdom) (entity_at princess house) (entity_at brother_thief house) (entity_at brother_astronomer house) (entity_at brother_hunter house) (entity_at brother_tailor house))
   )
)