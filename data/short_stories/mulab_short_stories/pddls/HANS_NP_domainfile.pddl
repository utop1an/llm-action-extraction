(define (domain Hansel_And_Grettel)
   (:requirements
      :conditional-effects :negative-preconditions :strips :typing)

   (:types 
      entity - object
      item - object
      location - object
   )

   (:constants 
      apple - item
      bird - entity
      branch - location
      bread_piece - item
      brushwood - item
      cake - item
      cottage - location
      crumb - item
      dead_tree - location
      door - location
      duck - entity
      father - entity
      father_house - location
      forest - location
      grettel - entity
      hansel - entity
      hill - location
      kettle - item
      lake - location
      milk - item
      mother - entity
      nut - item
      oven - item
      pancake - item
      path - location
      pearl - item
      pebble - item
      precious_stone - item
      roof - location
      room - location
      stable_location - location
      sugar_window - item
      window - location
      witch - entity
   )

   (:predicates 
      (at ?e - entity ?l - location)
      (bird_heard ?c - entity ?l - location)
      (bread_given ?c - entity)
      (captured ?c - entity)
      (collected ?c - entity ?i - item)
      (crossed_lake ?c - entity)
      (crumb_on_path ?l - location)
      (dead ?e - entity)
      (duck_helped ?c - entity ?d - entity)
      (fire_lit ?l - location)
      (has ?e - entity ?i - item)
      (locked ?l - location)
      (lost ?c - entity)
      (pebble_on_path ?l - location)
      (reunited_with_father ?c - entity)
   )

   (:action abandon_children_in_forest
     :parameters (?father ?g ?h ?mother - entity ?forest - location)
     :effect (and (at ?h ?forest) (at ?g ?forest) (lost ?h) (lost ?g) (bread_given ?h) (bread_given ?g) (fire_lit ?forest))
   )
   
   (:action leave_children_at_fire
     :parameters (?father ?g ?h ?mother - entity ?forest - location)
     :precondition (and (at ?h ?forest) (at ?g ?forest) (fire_lit ?forest) (bread_given ?h) (bread_given ?g))
     :effect (and (lost ?h) (lost ?g))
   )
   
   (:action drop_pebble_on_path
     :parameters (?h - entity ?peb - item ?path_loc - location)
     :precondition (and (at ?h ?path_loc) (has ?h ?peb))
     :effect (and (pebble_on_path ?path_loc) (not (has ?h ?peb)))
   )
   
   (:action follow_pebbles_home
     :parameters (?g ?h - entity ?home ?path_loc - location)
     :precondition (and (pebble_on_path ?path_loc) (at ?h ?path_loc) (at ?g ?path_loc))
     :effect (and (at ?h ?home) (at ?g ?home) (not (at ?h ?path_loc)) (not (at ?g ?path_loc)) (not (lost ?h)) (not (lost ?g)))
   )
   
   (:action drop_breadcrumb_on_path
     :parameters (?h - entity ?crumb - item ?path_loc - location)
     :precondition (and (at ?h ?path_loc) (has ?h ?crumb))
     :effect (and (crumb_on_path ?path_loc) (not (has ?h ?crumb)))
   )
   
   (:action follow_breadcrumbs_home
     :parameters (?g ?h - entity ?home ?path_loc - location)
     :precondition (and (at ?h ?path_loc) (at ?g ?path_loc))
     :effect (when (crumb_on_path ?path_loc) (and (at ?h ?home) (at ?g ?home) (not (at ?h ?path_loc)) (not (at ?g ?path_loc)) (not (lost ?h)) (not (lost ?g))))
   )
   
   (:action wander_forest_seeking_way_out
     :parameters (?g ?h - entity ?forest - location)
     :precondition (and (at ?h ?forest) (at ?g ?forest))
     :effect (and (lost ?h) (lost ?g))
   )
   
   (:action hear_bird_and_follow_to_witch_house
     :parameters (?bird ?g ?h - entity ?branch ?cottage - location)
     :precondition (and (at ?h ?branch) (at ?g ?branch) (bird_heard ?h ?branch) (bird_heard ?g ?branch))
     :effect (and (at ?h ?cottage) (at ?g ?cottage) (not (at ?h ?branch)) (not (at ?g ?branch)))
   )
   
   (:action enter_witch_house
     :parameters (?g ?h - entity ?cottage - location)
     :precondition (and (at ?h ?cottage) (at ?g ?cottage))
     :effect (and (has ?h cake) (has ?g sugar_window))
   )
   
   (:action witch_captures_children
     :parameters (?g ?h ?witch - entity ?cottage ?stable - location)
     :precondition (and (at ?h ?cottage) (at ?g ?cottage) (not (locked ?stable)))
     :effect (and (captured ?h) (locked ?stable) (at ?h ?stable) (not (at ?h ?cottage)))
   )
   
   (:action push_witch_into_oven
     :parameters (?g ?witch - entity ?oven - item ?room - location)
     :precondition (and (at ?g ?room) (at ?witch ?room))
     :effect (and (dead ?witch) (not (at ?witch ?room)))
   )
   
   (:action escape_from_stable
     :parameters (?g ?h - entity ?cottage ?stable - location)
     :precondition (and (captured ?h) (locked ?stable) (at ?g ?cottage) (at ?h ?stable))
     :effect (and (at ?h ?cottage) (not (at ?h ?stable)) (not (locked ?stable)) (not (captured ?h)))
   )
   
   (:action collect_treasure
     :parameters (?g ?h - entity ?pearl ?precious_stone - item ?cottage - location)
     :precondition (and (at ?h ?cottage) (at ?g ?cottage))
     :effect (and (has ?h ?pearl) (has ?g ?precious_stone) (collected ?h ?pearl) (collected ?g ?precious_stone))
   )
   
   (:action call_duck_for_crossing
     :parameters (?duck ?g ?h - entity ?lake - location)
     :precondition (and (at ?h ?lake) (at ?g ?lake) (at ?duck ?lake))
     :effect (and (duck_helped ?h ?duck) (duck_helped ?g ?duck) (crossed_lake ?h) (crossed_lake ?g))
   )
   
   (:action return_home_and_reunite_with_father
     :parameters (?father ?g ?h - entity ?father_house - location)
     :precondition (and (crossed_lake ?h) (crossed_lake ?g) (at ?father ?father_house))
     :effect (and (at ?h ?father_house) (at ?g ?father_house) (reunited_with_father ?h) (reunited_with_father ?g) (not (lost ?h)) (not (lost ?g)))
   )
)