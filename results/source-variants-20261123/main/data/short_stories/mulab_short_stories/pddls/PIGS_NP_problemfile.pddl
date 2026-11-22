(define (problem The_Three_Little_Pigs_problem)
   (:domain The_Three_Little_Pigs)


   (:init 
      (at mother_pig pig_home)
      (at pig1 pig_home)
      (at pig2 pig_home)
      (at pig3 pig_home)
      (at wolf oak_forest)
      (alive mother_pig)
      (alive pig1)
      (alive pig2)
      (alive pig3)
      (alive wolf)
      (has_bundle man_straw straw_bundle)
      (has_bundle man_furze furze_bundle)
      (has_bundle man_bricks bricks)
      (has_bundle man_bricks mortar)
      (has_bundle man_bricks trowel)
   )

   (:goal 
      (and (dead wolf) (alive pig3) (house_intact brick_house))
   )
)