import torch


def winston_loss(output, mu=1, sum=True):

	# MAIN CLASSES
	ALBATROSS = output[:, 0]
	CHEETAH = output[:, 1]
	GIRAFFE = output[:, 2]
	OSTRICH = output[:, 3]
	PENGUIN = output[:, 4]
	TIGER = output[:, 5]
	ZEBRA = output[:, 6]

	# ATTRIBUTE CLASSES
	BIRD = output[:, 7]
	BLACK = output[:, 8]
	BLACKSTRIPES = output[:, 9]
	BLACKWHITE = output[:, 10]
	CARNIVORE = output[:, 11]
	CLAWS = output[:, 12]
	CUD = output[:, 13]
	DARKSPOTS = output[:, 14]
	EVENTOED = output[:, 15]
	FEATHER = output[:, 16]
	FLY = output[:, 17]
	FORWARDEYES = output[:, 18]
	GOODFLIER = output[:, 19]
	HAIR = output[:, 20]
	HOOFS = output[:, 21]
	LAYEGGS = output[:, 22]
	LONGLEGS = output[:, 23]
	LONGNECK = output[:, 24]
	MAMMAL = output[:, 25]
	MEAT = output[:, 26]
	MILK = output[:, 27]
	POINTEDTEETH = output[:, 28]
	SWIM = output[:, 29]
	TAWNY = output[:, 30]
	UNGULATE = output[:, 31]
	WHITE = output[:, 32]

	# here we converted each FOL rule using the product T-Norm (no-residual)
	loss_fol_product_tnorm = [
		# 0) HAIR => MAMMAL
		(HAIR * (1. - MAMMAL)),
		# 1) MILK => MAMMAL
		(MILK * (1. - MAMMAL)),
		# 2) FEATHER => BIRD
		(FEATHER * (1. - BIRD)),
		# 3) FLY ^ LAYEGGS => BIRD
		((FLY * LAYEGGS) * (1. - BIRD)),
		# 4) MAMMAL ^ MEAT => CARNIVORE
		((MAMMAL * MEAT) * (1. - CARNIVORE)),
		# 5) MAMMAL ^ POINTEDTEETH ^ CLAWS ^ FORWARDEYES => CARNIVORE
		((MAMMAL * POINTEDTEETH * CLAWS * FORWARDEYES) * (1. - CARNIVORE)),
		# 6) MAMMAL ^ HOOFS => UNGULATE
		((MAMMAL * HOOFS) * (1. - UNGULATE)),
		# 7) MAMMAL ^ CUD => UNGULATE
		((MAMMAL * CUD) * (1. - UNGULATE)),
		# 8) MAMMAL ^ CUD => EVENTOED
		((MAMMAL * CUD) * (1. - EVENTOED)),
		# 9)CARNIVORE ^ TAWNY ^ DARKSPOTS => CHEETAH
		((CARNIVORE * TAWNY * DARKSPOTS) * (1. - CHEETAH)),
		# 10)CARNIVORE ^ TAWNY ^ BLACKWHITE => TIGER
		((CARNIVORE * TAWNY * BLACKWHITE) * (1. - TIGER)),
		# 11) UNGULATE ^ LONGLEGS ^ LONGNECK ^ TAWNY ^ DARKSPOTS => GIRAFFE
		((UNGULATE * LONGLEGS * LONGNECK * TAWNY * DARKSPOTS) * (1. - GIRAFFE)),
		# 12) BLACKSTRIPES ^ UNGULATE ^ WHITE => ZEBRA
		((BLACKSTRIPES * UNGULATE * WHITE) * (1. - ZEBRA)),
		# 13) BIRD ^ !FLY ^ LONGLEGS ^ LONGNECK ^ BLACK => OSTRICH
		((BIRD * (1. - FLY) * LONGLEGS * LONGNECK * BLACK) * (1. - OSTRICH)),
		# 14) BIRD ^ !FLY ^ LONGLEGS ^ SWIM ^ BLACKWHITE => PENGUIN
		((BIRD * (1. - FLY) * SWIM * BLACKWHITE) * (1. - PENGUIN)),
		# 15) BIRD ^ GOODFLIER => ALBATROSS
		((BIRD * GOODFLIER) * (1. - ALBATROSS)),

		# 16) XOR ON THE MAIN CLASSES
		mu * ((1 - ((ALBATROSS) * (1 - CHEETAH) * (1 - GIRAFFE) * (1 - OSTRICH) * (1 - PENGUIN) * (1 - TIGER) * (1 - ZEBRA))) *
		      (1 - ((1 - ALBATROSS) * (CHEETAH) * (1 - GIRAFFE) * (1 - OSTRICH) * (1 - PENGUIN) * (1 - TIGER) * (1 - ZEBRA))) *
		      (1 - ((1 - ALBATROSS) * (1 - CHEETAH) * (GIRAFFE) * (1 - OSTRICH) * (1 - PENGUIN) * (1 - TIGER) * (1 - ZEBRA))) *
		      (1 - ((1 - ALBATROSS) * (1 - CHEETAH) * (1 - GIRAFFE) * (OSTRICH) * (1 - PENGUIN) * (1 - TIGER) * (1 - ZEBRA))) *
		      (1 - ((1 - ALBATROSS) * (1 - CHEETAH) * (1 - GIRAFFE) * (1 - OSTRICH) * (PENGUIN) * (1 - TIGER) * (1 - ZEBRA))) *
		      (1 - ((1 - ALBATROSS) * (1 - CHEETAH) * (1 - GIRAFFE) * (1 - OSTRICH) * (1 - PENGUIN) * (TIGER) * (1 - ZEBRA))) *
		      (1 - ((1 - ALBATROSS) * (1 - CHEETAH) * (1 - GIRAFFE) * (1 - OSTRICH) * (1 - PENGUIN) * (1 - TIGER) * (ZEBRA)))),

		# 17) OR ON THE ATTRIBUTE CLASSES
		mu*((1 - BIRD) * (1 - BLACK) * (1 - BLACKSTRIPES) * (1 - BLACKWHITE) * (1 - CARNIVORE) * (1 - CLAWS) * (1 - CUD) *
		        (1 - DARKSPOTS) * (1 - EVENTOED) * (1 - FEATHER) * (1 - FLY) * (1 - FORWARDEYES) * (
				 1 - GOODFLIER) * (1 - HAIR) * (1 - HOOFS) * (1 - LAYEGGS) * (1 - LONGLEGS) * (1 - LONGNECK) * (
				 1 - MAMMAL) * (1 - MEAT) * (1 - MILK) * (1 - POINTEDTEETH) * (1 - SWIM) * (1 - TAWNY) * (
				 1 - UNGULATE) * (1 - WHITE))
	]

	if sum:
		losses = torch.sum(torch.stack(loss_fol_product_tnorm, dim=0), dim=1)
	else:
		losses = torch.stack(loss_fol_product_tnorm, dim=0)

	loss_sum = torch.squeeze(torch.sum(losses, dim=0))

	return loss_sum
