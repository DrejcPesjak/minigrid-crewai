(define (problem letseat-simple)
	(:domain letseat)
	(:objects
	arm - robot
	cupcake1 - cupcake
	table - location
	plate - location
	)

	(:init
		(on arm table)
		(on cupcake1 table)
		(arm-empty)
		(path table plate)
	)
	(:goal 
		(on cupcake1 plate)
	)
)