    (define (problem BLOCKS-1-3) (:domain blocks)
(:objects
	a - block
	c - block
	b - block
	d - block
    a1 - agent
    a2 - agent
    a3 - agent
    big - size
    small - size
    middle - size
)
(:init
	(handempty a1)
	(handempty a2)
	(handempty a3)
	(clear a)
	(clear d)
	(ontable c)
	(ontable d)
	(on a b)
	(on b c)
	(blocktype middle a)
	(blocktype small b)
	(blocktype big c)
	(blocktype small d)
)
(:goal
	(and
	    (handempty a1)
	    (handempty a2)
	    (handempty a3)
	    (clear d)
		(on d c)
		(on c b)
		(on b a)
		(ontable a)
        (blocktype middle a)
        (blocktype small b)
        (blocktype big c)
        (blocktype small d)
	)
)


(:constraints
    (and
        (and (always (forall (?x - block)
            (implies (or (blocktype big ?x) (blocktype middle ?x)) (holding a3 ?x))))
        )
        (and (always (forall (?x - block)
            (implies (or (blocktype big ?x) (blocktype small ?x)) (holding a1 ?x))))
        )
        (and (always (forall (?x - block)
            (implies (or (blocktype middle ?x) (blocktype small ?x)) (holding a2 ?x))))
        )
    )
)
)



