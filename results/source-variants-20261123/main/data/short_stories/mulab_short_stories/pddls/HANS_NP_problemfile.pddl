(define (problem Hansel_And_Grettel_problem)
   (:domain Hansel_And_Grettel)


   (:init 
      (at father father_house)
      (at hansel father_house)
      (at grettel father_house)
      (at mother father_house)
      (has hansel pebble)
   )

   (:goal 
      (and (at hansel father_house) (at grettel father_house) (reunited_with_father hansel) (reunited_with_father grettel) (dead witch) (has hansel pearl) (has grettel precious_stone) (collected hansel pearl) (collected grettel precious_stone))
   )
)