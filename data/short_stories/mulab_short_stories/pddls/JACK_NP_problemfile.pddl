(define (problem Jack_And_The_Beanstalk_problem)
   (:domain Jack_And_The_Beanstalk)


   (:init 
      (at jack cottage)
      (at jack_mother cottage)
      (at milky_white cottage)
      (at butcher market)
      (at giant giant_castle)
      (at giant_wife giant_castle)
      (has jack_mother axe)
      (offers butcher magic_beans)
      (at_item magic_beans market)
      (at_item gold_bag_1 giant_castle)
      (at_item gold_bag_2 giant_castle)
      (at_item axe cottage)
      (at_item oak_tree_club giant_castle)
      (on golden_harp giant_castle)
   )

   (:goal 
      (and (has jack gold_bag_1) (has jack gold_bag_2) (has jack golden_harp) (dead giant) (at jack cottage) (at jack_mother cottage) (not (intact beanstalk)))
   )
)