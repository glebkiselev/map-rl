(define (problem BLOCKS-1-0) (:domain blocks)
(:objects
	block-c - block
	block-d - block
    ag1 - agent
    big - size
)
(:init
	(handempty ag1)
	(clear block-c)
	(clear block-d)
	(onground block-c)
	(onground block-d)
	(blocktype big block-c)
	(blocktype big block-d)
)
(:goal
	(and
	    (handempty ag1)
		(on block-c block-d)
        (blocktype big block-c)
        (blocktype big block-d)
        (clear block-c)
        (onground block-d)
	)
)

(:constraints
    (and

        (and (always (forall (?x - block)
            (implies (blocktype big ?x)(holding ag1 ?x))))
        )

    )
)
)
