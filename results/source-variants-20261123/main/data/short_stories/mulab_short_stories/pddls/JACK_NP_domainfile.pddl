(define (domain Jack_And_The_Beanstalk)
   (:requirements
      :strips :typing)

   (:types 
      entity - object
      item - object
      location - object
   )

   (:constants 
      axe - item
      beanstalk - location
      brown_hen - entity
      butcher - entity
      castle_road - location
      cottage - location
      fairy_maiden - entity
      garden - location
      giant - entity
      giant_castle - location
      giant_dog - entity
      giant_wife - entity
      gold_bag_1 - item
      gold_bag_2 - item
      golden_harp - item
      jack - entity
      jack_mother - entity
      kettle - location
      magic_beans - item
      market - location
      milky_white - entity
      oak_tree_club - item
      oven - location
      window - location
   )

   (:predicates 
      (asleep ?e - entity)
      (at ?e - entity ?l - location)
      (at_item ?i - item ?l - location)
      (beanstalk_grown )
      (climbing_down ?e - entity)
      (dead ?e - entity)
      (distracted ?e - entity)
      (empty ?i - item)
      (has ?e - entity ?i - item)
      (has_breakfast ?e - entity)
      (inside ?e - entity ?i - item)
      (intact ?l - location)
      (offers ?e - entity ?i - item)
      (on ?i - item ?l - location)
      (planted ?i - item)
   )

   (:action sell_cow_for_beans
     :parameters (?b ?cow ?j - entity ?beans - item ?l - location)
     :precondition (and (at ?j ?l) (at ?cow ?l) (offers ?b ?beans) (at ?b ?l))
     :effect (and (has ?j ?beans) (not (offers ?b ?beans)) (not (at_item ?beans ?l)))
   )
   
   (:action plant_beans
     :parameters (?beans - item ?loc - location)
     :precondition (at_item ?beans ?loc)
     :effect (and (planted ?beans) (not (at_item ?beans ?loc)))
   )
   
   (:action grow_beanstalk
     :parameters (?beans - item)
     :precondition (planted ?beans)
     :effect (and (beanstalk_grown) (intact beanstalk))
   )
   
   (:action climb_beanstalk
     :parameters (?j - entity)
     :precondition (and (beanstalk_grown) (at ?j beanstalk))
     :effect (and (at ?j giant_castle) (not (at ?j beanstalk)))
   )
   
   (:action ask_giant_wife_for_breakfast
     :parameters (?j ?wife - entity)
     :precondition (and (at ?j giant_castle) (at ?wife giant_castle))
     :effect (has_breakfast ?j)
   )
   
   (:action hide_in_kettle
     :parameters (?j - entity ?kettle - item)
     :precondition (and (at ?j giant_castle) (empty ?kettle))
     :effect (and (inside ?j ?kettle) (not (at ?j giant_castle)))
   )
   
   (:action steal_gold_bags
     :parameters (?j - entity ?bag1 ?bag2 - item)
     :precondition (and (asleep giant) (at_item ?bag1 giant_castle) (at_item ?bag2 giant_castle))
     :effect (and (has ?j ?bag1) (has ?j ?bag2) (not (at_item ?bag1 giant_castle)) (not (at_item ?bag2 giant_castle)))
   )
   
   (:action return_down_beanstalk
     :parameters (?j - entity)
     :precondition (at ?j giant_castle)
     :effect (and (at ?j cottage) (not (at ?j giant_castle)))
   )
   
   (:action hide_in_oven
     :parameters (?j - entity ?oven - item)
     :precondition (at ?j giant_castle)
     :effect (and (inside ?j ?oven) (not (at ?j giant_castle)))
   )
   
   (:action steal_hen
     :parameters (?giant ?j - entity ?hen - item)
     :precondition (and (on ?hen giant_castle) (distracted ?giant))
     :effect (and (has ?j ?hen) (not (on ?hen giant_castle)))
   )
   
   (:action steal_harp
     :parameters (?giant ?j - entity ?harp - item)
     :precondition (and (on ?harp giant_castle) (asleep ?giant))
     :effect (and (has ?j ?harp) (not (on ?harp giant_castle)))
   )
   
   (:action cut_beanstalk
     :parameters (?giant ?mother - entity ?axe - item)
     :precondition (and (intact beanstalk) (climbing_down ?giant))
     :effect (and (dead ?giant) (not (intact beanstalk)))
   )
)