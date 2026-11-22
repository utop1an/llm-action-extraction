(define (domain The_Four_Skilful_Brothers)
   (:requirements
      :strips :typing)

   (:types 
      entity - object
      item - object
      location - object
   )

   (:constants 
      brother_astronomer - entity
      brother_hunter - entity
      brother_tailor - entity
      brother_thief - entity
      chaffinch - entity
      cross_way - location
      dragon - entity
      eggs - item
      father - entity
      gun - item
      house - location
      king - entity
      kingdom - location
      mentor_astronomer - entity
      mentor_hunter - entity
      mentor_tailor - entity
      mentor_thief - entity
      needle - item
      planks - item
      princess - entity
      rock_sea - location
      sea - location
      ship - item
      sticks - item
      telescope - item
      town_gate - location
      tree - location
   )

   (:predicates 
      (dragon_dead )
      (eggs_fetched ?e - entity)
      (eggs_sewn ?e - entity)
      (eggs_shot ?e - entity)
      (entity_at ?e - entity ?l - location)
      (half_kingdom ?e - entity ?k - location)
      (has ?e - entity ?i - item)
      (is_astronomer ?e - entity)
      (is_hunter ?e - entity)
      (is_tailor ?e - entity)
      (is_thief ?e - entity)
      (item_at ?i - item ?l - location)
      (princess_rescued ?e - entity)
      (reward_received ?e - entity)
      (ship_broken )
      (ship_repaired )
   )

   (:action leave_home
     :parameters (?b - entity)
     :precondition (entity_at ?b house)
     :effect (and (entity_at ?b town_gate) (not (entity_at ?b house)))
   )
   
   (:action separate_at_crossroads
     :parameters (?b - entity)
     :precondition (entity_at ?b cross_way)
     :effect (has ?b sticks)
   )
   
   (:action learn_thief_trade
     :parameters (?b - entity)
     :precondition (entity_at ?b cross_way)
     :effect (is_thief ?b)
   )
   
   (:action learn_astronomer_trade
     :parameters (?b - entity)
     :precondition (entity_at ?b cross_way)
     :effect (and (is_astronomer ?b) (has ?b telescope))
   )
   
   (:action learn_hunter_trade
     :parameters (?b - entity)
     :precondition (entity_at ?b cross_way)
     :effect (and (is_hunter ?b) (has ?b gun))
   )
   
   (:action learn_tailor_trade
     :parameters (?b - entity)
     :precondition (entity_at ?b cross_way)
     :effect (and (is_tailor ?b) (has ?b needle))
   )
   
   (:action reunite_at_crossroads
     :parameters (?b - entity)
     :precondition (entity_at ?b cross_way)
     :effect (and (entity_at ?b house) (not (entity_at ?b cross_way)))
   )
   
   (:action count_eggs
     :parameters (?a - entity)
     :precondition (and (is_astronomer ?a) (has ?a telescope) (entity_at ?a tree))
     :effect (has ?a eggs)
   )
   
   (:action fetch_eggs
     :parameters (?t - entity)
     :precondition (and (is_thief ?t) (entity_at ?t tree) (item_at eggs tree))
     :effect (and (eggs_fetched ?t) (has ?t eggs) (not (item_at eggs tree)))
   )
   
   (:action shoot_eggs
     :parameters (?h - entity)
     :precondition (and (is_hunter ?h) (has ?h gun) (eggs_fetched brother_thief))
     :effect (eggs_shot ?h)
   )
   
   (:action sew_eggs
     :parameters (?t - entity)
     :precondition (and (is_tailor ?t) (has ?t needle) (eggs_shot brother_hunter))
     :effect (eggs_sewn ?t)
   )
   
   (:action return_eggs_to_nest
     :parameters (?t - entity)
     :precondition (and (is_thief ?t) (eggs_sewn brother_tailor) (entity_at ?t tree) (has ?t eggs))
     :effect (and (item_at eggs tree) (not (has ?t eggs)))
   )
   
   (:action locate_princess
     :parameters (?a - entity)
     :precondition (and (is_astronomer ?a) (has ?a telescope))
     :effect (entity_at princess rock_sea)
   )
   
   (:action obtain_ship
     :parameters (?a - entity)
     :precondition (is_astronomer ?a)
     :effect (has ?a ship)
   )
   
   (:action sail_to_rock
     :parameters (?a - entity)
     :precondition (and (has ?a ship) (entity_at ?a house))
     :effect (and (entity_at ?a rock_sea) (not (entity_at ?a house)))
   )
   
   (:action steal_princess
     :parameters (?t - entity)
     :precondition (and (is_thief ?t) (entity_at ?t rock_sea) (entity_at princess rock_sea) (entity_at dragon rock_sea))
     :effect (and (princess_rescued ?t) (entity_at princess sea) (not (entity_at princess rock_sea)))
   )
   
   (:action shoot_dragon
     :parameters (?h - entity)
     :precondition (and (is_hunter ?h) (has ?h gun) (entity_at dragon rock_sea))
     :effect (and (dragon_dead) (ship_broken) (not (entity_at dragon rock_sea)))
   )
   
   (:action swim_after_ship_break
     :parameters (?b - entity)
     :precondition (ship_broken)
     :effect (entity_at ?b sea)
   )
   
   (:action repair_ship
     :parameters (?t - entity)
     :precondition (and (is_tailor ?t) (has ?t needle) (ship_broken))
     :effect (and (ship_repaired) (not (ship_broken)))
   )
   
   (:action return_home_with_princess
     :parameters (?a - entity)
     :precondition (and (ship_repaired) (entity_at ?a sea))
     :effect (and (entity_at ?a house) (entity_at princess house) (not (entity_at ?a sea)))
   )
   
   (:action receive_reward
     :parameters (?e - entity)
     :precondition (princess_rescued ?e)
     :effect (and (reward_received ?e) (half_kingdom ?e kingdom))
   )
)