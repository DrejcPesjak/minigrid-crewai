(define (domain cupcake-domain)
  (:requirements :strips :typing)
  (:types robot cupcake-type plate-type)

  (:predicates
    (empty ?r - robot)
    (holding ?r - robot ?c - cupcake-type)
    (on-table ?c - cupcake-type)
    (on-plate ?c - cupcake-type)
  )

  (:action pick-up
    :parameters (?r - robot ?c - cupcake-type)
    :precondition (and (empty ?r) (on-table ?c))
    :effect (and (holding ?r ?c)
                 (not (empty ?r))
                 (not (on-table ?c)))
  )

  (:action put-on-plate
    :parameters (?r - robot ?c - cupcake-type)
    :precondition (and (holding ?r ?c))
    :effect (and (on-plate ?c)
                 (empty ?r)
                 (not (holding ?r ?c))))
)