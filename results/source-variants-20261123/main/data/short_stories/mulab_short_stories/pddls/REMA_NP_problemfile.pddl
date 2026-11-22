(define (problem The_Three_Remarks_problem)
   (:domain The_Three_Remarks)


   (:init 
      (at princess back_stairs)
      (at impostor_butterman market_place)
      (at robber_chief forest)
      (at king_at_palace palace)
      (has impostor_butterman bag_of_rubies)
      (has robber_chief bag_of_sapphires)
      (has impostor_butterman sceptre)
   )

   (:goal 
      (and (married princess king_at_palace) (with_all_my_heart_spoken))
   )
)