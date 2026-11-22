(define (domain The_Three_Remarks)
   (:requirements
      :negative-preconditions :strips :typing)

   (:types 
      entity - object
      item - object
      location - object
   )

   (:constants 
      back_door - location
      back_stairs - location
      bag_of_rubies - item
      bag_of_sapphires - item
      city - location
      crown - item
      father_king - entity
      forest - location
      impostor_butterman - entity
      king_at_palace - entity
      market_place - location
      oak_tree - location
      palace - location
      princess - entity
      robber_chief - entity
      sceptre - item
   )

   (:predicates 
      (at ?e - entity ?l - location)
      (grandmother_mangle_spoken )
      (has ?e - entity ?i - item)
      (married ?p1 - entity ?p2 - entity)
      (price_of_butter_spoken )
      (with_all_my_heart_spoken )
   )

   (:action run_away_from_palace
     :parameters (?p - entity)
     :precondition (at ?p back_stairs)
     :effect (and (at ?p city) (not (at ?p back_stairs)))
   )
   
   (:action encounter_impostor_king
     :parameters (?i ?p - entity)
     :precondition (and (at ?p city) (at ?i market_place))
     :effect (and (at ?p market_place) (not (at ?p city)))
   )
   
   (:action utter_phrase_price_of_butter
     :parameters (?p - entity)
     :precondition (and (at ?p market_place) (not (price_of_butter_spoken)))
     :effect (price_of_butter_spoken)
   )
   
   (:action receive_rubies_from_impostor
     :parameters (?i ?p - entity ?r - item)
     :precondition (and (price_of_butter_spoken) (at ?p market_place) (has ?i ?r))
     :effect (and (has ?p ?r) (not (has ?i ?r)))
   )
   
   (:action encounter_robbers_in_forest
     :parameters (?p - entity)
     :precondition (at ?p forest)
     :effect (and (at ?p oak_tree) (not (at ?p forest)))
   )
   
   (:action utter_phrase_grandmother_mangle
     :parameters (?p - entity)
     :precondition (and (at ?p oak_tree) (not (grandmother_mangle_spoken)))
     :effect (grandmother_mangle_spoken)
   )
   
   (:action receive_sapphires_from_robber_chief
     :parameters (?c ?p - entity ?s - item)
     :precondition (and (grandmother_mangle_spoken) (at ?p oak_tree) (has ?c ?s))
     :effect (and (has ?p ?s) (not (has ?c ?s)))
   )
   
   (:action arrive_at_marble_palace
     :parameters (?p - entity)
     :precondition (at ?p oak_tree)
     :effect (and (at ?p palace) (not (at ?p oak_tree)))
   )
   
   (:action king_requests_information_and_gifts
     :parameters (?k ?p - entity ?r ?s - item)
     :precondition (and (at ?p palace) (has ?p ?r) (has ?p ?s) (at ?k palace))
     :effect (and (has ?k ?r) (has ?k ?s) (not (has ?p ?r)) (not (has ?p ?s)))
   )
   
   (:action king_proposes_marriage
     :parameters (?k ?p - entity)
     :precondition (and (at ?p palace) (at ?k palace))
     :effect (married ?p ?k)
   )
   
   (:action princess_accepts_with_heart
     :parameters (?k ?p - entity)
     :precondition (and (married ?p ?k) (not (with_all_my_heart_spoken)))
     :effect (with_all_my_heart_spoken)
   )
   
   (:action suitors_retire_due_to_phrase
     :parameters (?s - entity)
     :precondition (and (grandmother_mangle_spoken) (at ?s palace))
     :effect (not (married ?s princess))
   )
)