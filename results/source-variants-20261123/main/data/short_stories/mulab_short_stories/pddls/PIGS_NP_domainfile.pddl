(define (domain The_Three_Little_Pigs)
   (:requirements
      :strips :typing)

   (:types 
      entity - object
      item - object
      location - object
   )

   (:constants 
      apple - item
      brick_house - location
      bricks - item
      butter_churn - item
      chimney - location
      farmer - entity
      fire - item
      furze_bundle - item
      furze_house - location
      hearth - location
      hill - location
      man_bricks - entity
      man_furze - entity
      man_straw - entity
      mortar - item
      mother_pig - entity
      oak_forest - location
      pig1 - entity
      pig2 - entity
      pig3 - entity
      pig_home - location
      pot_of_water - item
      sack - item
      squire - entity
      squire_orchard - location
      straw_bundle - item
      straw_house - location
      town_fair - location
      trowel - item
      turnip_field - location
      turnips_load - item
      wolf - entity
   )

   (:predicates 
      (alive ?e - entity)
      (at ?e - entity ?l - location)
      (boiling ?i - item)
      (dead ?e - entity)
      (door_closed ?h - location)
      (entry_denied ?w - entity ?h - location)
      (entry_requested ?w - entity ?h - location)
      (escaped ?e - entity)
      (has ?e - entity ?i - item)
      (has_bundle ?m - entity ?i - item)
      (house ?h - location ?m - item)
      (house_destroyed ?h - location)
      (house_intact ?h - location)
      (huffed ?w - entity)
      (inside ?e - entity ?c - item)
      (material_requested ?p - entity ?i - item)
      (over ?i1 - item ?i2 - item)
      (plan ?e - entity ?l - location)
      (puffed ?w - entity)
      (rolling ?i - item)
      (under ?i - item ?l - location)
   )

   (:action send_sons_on_journey
     :parameters (?m ?p - entity ?d - location)
     :precondition (and (at ?m pig_home) (at ?p pig_home) (alive ?m) (alive ?p))
     :effect (and (plan ?p ?d) (not (at ?p pig_home)))
   )
   
   (:action ask_for_material
     :parameters (?m ?p - entity ?i - item)
     :precondition (and (has_bundle ?m ?i) (alive ?p) (alive ?m))
     :effect (material_requested ?p ?i)
   )
   
   (:action receive_material
     :parameters (?m ?p - entity ?i - item)
     :precondition (and (material_requested ?p ?i) (has_bundle ?m ?i) (alive ?p) (alive ?m))
     :effect (and (has ?p ?i) (not (material_requested ?p ?i)) (not (has_bundle ?m ?i)))
   )
   
   (:action build_house
     :parameters (?p - entity ?i - item ?h - location)
     :precondition (and (has ?p ?i) (alive ?p) (at ?p ?h))
     :effect (and (house ?h ?i) (house_intact ?h) (door_closed ?h) (not (has ?p ?i)))
   )
   
   (:action wolf_knocks_and_requests_entry
     :parameters (?w - entity ?h - location)
     :precondition (and (at ?w ?h) (door_closed ?h) (alive ?w))
     :effect (entry_requested ?w ?h)
   )
   
   (:action pig_refuses_entry
     :parameters (?w - entity ?h - location)
     :precondition (and (entry_requested ?w ?h) (alive ?w))
     :effect (and (entry_denied ?w ?h) (not (entry_requested ?w ?h)))
   )
   
   (:action wolf_huff_and_puff
     :parameters (?w - entity ?h - location)
     :precondition (and (entry_denied ?w ?h) (alive ?w))
     :effect (and (huffed ?w) (puffed ?w))
   )
   
   (:action wolf_blows_house_down
     :parameters (?w - entity ?i - item ?h - location)
     :precondition (and (huffed ?w) (puffed ?w) (house ?h ?i) (door_closed ?h) (alive ?w))
     :effect (and (house_destroyed ?h) (not (house_intact ?h)) (not (door_closed ?h)))
   )
   
   (:action wolf_eats_pig
     :parameters (?p ?w - entity ?h - location)
     :precondition (and (house_destroyed ?h) (at ?p ?h) (alive ?p) (alive ?w))
     :effect (and (dead ?p) (not (alive ?p)) (not (at ?p ?h)))
   )
   
   (:action wolf_fails_to_blow_house
     :parameters (?w - entity ?i - item ?h - location)
     :precondition (and (huffed ?w) (puffed ?w) (house ?h ?i) (alive ?w))
     :effect (house_intact ?h)
   )
   
   (:action wolf_promises_turnip_field
     :parameters (?p ?w - entity ?l - location)
     :precondition (and (alive ?w) (alive ?p))
     :effect (plan ?p ?l)
   )
   
   (:action pig_collects_turnips
     :parameters (?p - entity ?i - item ?l - location)
     :precondition (and (plan ?p ?l) (alive ?p))
     :effect (and (has ?p ?i) (not (plan ?p ?l)))
   )
   
   (:action wolf_promises_apple_orchard
     :parameters (?p ?w - entity ?l - location)
     :precondition (and (alive ?w) (alive ?p))
     :effect (plan ?p ?l)
   )
   
   (:action pig_collects_apples_and_escapes
     :parameters (?p - entity ?a - item ?l - location)
     :precondition (and (plan ?p ?l) (alive ?p))
     :effect (and (has ?p ?a) (escaped ?p) (not (plan ?p ?l)))
   )
   
   (:action wolf_promises_fair_trip
     :parameters (?p ?w - entity ?l - location)
     :precondition (and (alive ?w) (alive ?p))
     :effect (plan ?p ?l)
   )
   
   (:action pig_buys_butter_churn_and_hides
     :parameters (?p - entity ?c - item)
     :precondition (and (at ?p town_fair) (alive ?p))
     :effect (and (has ?p ?c) (inside ?p ?c))
   )
   
   (:action pig_rolls_churn_down_hill
     :parameters (?p - entity ?c - item)
     :precondition (and (inside ?p ?c) (alive ?p))
     :effect (and (rolling ?c) (not (inside ?p ?c)))
   )
   
   (:action wolf_flees_from_rolling_object
     :parameters (?w - entity ?c - item)
     :precondition (and (rolling ?c) (alive ?w))
     :effect (escaped ?w)
   )
   
   (:action wolf_attempts_chimney_entry
     :parameters (?w - entity ?h - location)
     :precondition (and (house_intact ?h) (alive ?w))
     :effect (entry_requested ?w ?h)
   )
   
   (:action pig_prepares_boiling_pot_and_fire
     :parameters (?p - entity ?fire ?pot - item ?h - location)
     :precondition (and (at ?p ?h) (has ?p ?pot) (has ?p ?fire) (alive ?p))
     :effect (and (over ?pot ?fire) (under ?pot chimney) (boiling ?pot))
   )
   
   (:action wolf_falls_into_pot_and_is_boiled
     :parameters (?w - entity ?pot - item)
     :precondition (and (boiling ?pot) (alive ?w))
     :effect (and (dead ?w) (not (alive ?w)))
   )
)