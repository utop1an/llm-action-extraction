(define (problem Chicken-Licken_problem)
   (:domain Chicken-Licken)


   (:init 
      (at Chicken-licken Wood)
      (going_to_wood Chicken-licken)
      (sky_fell_on_head Chicken-licken)
      (at Hen-len Wood)
      (going_to_wood Hen-len)
      (at Cock-lock Wood)
      (going_to_wood Cock-lock)
      (at Duck-luck Wood)
      (going_to_wood Duck-luck)
      (at Drake-lake Wood)
      (going_to_wood Drake-lake)
      (at Goose-loose Wood)
      (going_to_wood Goose-loose)
      (at Gander-lander Wood)
      (going_to_wood Gander-lander)
      (at Turkey-lurkey Wood)
      (going_to_wood Turkey-lurkey)
      (at Fox-lox Wood)
   )

   (:goal 
      (and (eaten_by_fox Chicken-licken) (eaten_by_fox Hen-len) (eaten_by_fox Cock-lock) (eaten_by_fox Duck-luck) (eaten_by_fox Drake-lake) (eaten_by_fox Goose-loose) (eaten_by_fox Gander-lander) (eaten_by_fox Turkey-lurkey))
   )
)