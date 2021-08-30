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


def pascalpart_loss(output, sum=True):

	Aeroplane = output[:, 0]
	Aeroplane_Body = output[:, 1]
	Arm = output[:, 2]
	Backside = output[:, 3]
	Beak = output[:, 4]
	Bicycle = output[:, 5]
	Bird = output[:, 6]
	Boat = output[:, 7]
	Body = output[:, 8]  # wrong there are no "body" because have been separated into aeroplane_body and bott_body
	Bottle = output[:, 9]
	Bottle_Body = output[:, 10]
	Bus = output[:, 11]
	Cap = output[:, 12]
	Car = output[:, 13]
	Cat = output[:, 14]
	Chainwheel = output[:, 15]
	Chair = output[:, 16]
	Coach = output[:, 17]
	Cow = output[:, 18]
	Dog = output[:, 19]
	Door = output[:, 20]
	Ear = output[:, 21]
	Ebrow = output[:, 22]
	Engine = output[:, 23]
	Eye = output[:, 24]
	Foot = output[:, 25]
	Frontside = output[:, 26]
	Hair = output[:, 27]
	Hand = output[:, 28]
	Handlebar = output[:, 29]
	Head = output[:, 30]
	Headlight = output[:, 31]
	Hoof = output[:, 32]
	Horn = output[:, 33]
	Horse = output[:, 34]
	Leftside = output[:, 35]
	Leg = output[:, 36]
	Mirror = output[:, 37]
	Motorbike = output[:, 38]
	Mouth = output[:, 39]
	Muzzle = output[:, 40]
	Neck = output[:, 41]
	Nose = output[:, 42]
	Paw = output[:, 43]
	Person = output[:, 44]
	Plant = output[:, 45]
	Plate = output[:, 46]
	Pot = output[:, 47]
	Pottedplant = output[:, 48]
	Rightside = output[:, 49]
	Roofside = output[:, 50]
	Saddle = output[:, 51]
	Screen = output[:, 52]
	Sheep = output[:, 53]
	Sofa = output[:, 54]
	Stern = output[:, 55]
	Table = output[:, 56]
	Tail = output[:, 57]
	Torso = output[:, 58]
	Train = output[:, 59]
	Train_Head = output[:, 60]
	Tvmonitor = output[:, 61]
	Wheel = output[:, 62]
	Window = output[:, 63]
	Wing = output[:, 64]

	loss_fol_product_tnorm = [

		# A: OBJECT-PART --> [OBJECTS] RULES
		#  'Screen': ['Tvmonitor'],
		(Screen * (1 - Tvmonitor)),
		#  'Coach': ['Train'],
		(Coach * (1 - Train)),
		#  'Torso': ['Person', 'Horse', 'Cow', 'Dog', 'Bird', 'Cat', 'Sheep'],
		(Torso * (1 - Person) * (1 - Horse) * (1 - Cow) * (1 - Dog) * (1 - Bird) * (1 - Cat) * (1 - Sheep)),
		#  'Leg': ['Person', 'Horse', 'Cow', 'Dog', 'Bird', 'Cat', 'Sheep'],
		(Leg * (1 - Person) * (1 - Horse) * (1 - Cow) * (1 - Dog) * (1 - Bird) * (1 - Cat) * (1 - Sheep)),
		#  'Head': ['Person', 'Horse', 'Cow', 'Dog', 'Bird', 'Cat', 'Sheep'],
		(Head * (1 - Person) * (1 - Horse) * (1 - Cow) * (1 - Dog) * (1 - Bird) * (1 - Cat) * (1 - Sheep)),
		#  'Ear': ['Person', 'Horse', 'Cow', 'Dog', 'Cat', 'Sheep'],
		(Ear * (1 - Person) * (1 - Horse) * (1 - Cow) * (1 - Dog) * (1 - Cat) * (1 - Sheep)),
		#  'Eye': ['Person', 'Cow', 'Dog', 'Bird', 'Cat', 'Horse', 'Sheep'],
		(Eye * (1 - Person) * (1 - Cow) * (1 - Dog) * (1 - Bird) * (1 - Cat) * (1 - Horse) * (1 - Sheep)),
		#  'Ebrow': ['Person'],
		(Ebrow * (1 - Person)),
		#  'Mouth': ['Person'],
		(Mouth * (1 - Person)),
		#  'Hair': ['Person'],
		(Hair * (1 - Person)),
		#  'Nose': ['Person', 'Dog', 'Cat'],
		(Nose * (1 - Person) * (1 - Dog) * (1 - Cat)),
		#  'Neck': ['Person', 'Horse', 'Cow', 'Dog', 'Bird', 'Cat', 'Sheep'],
		(Neck * (1 - Person) * (1 - Horse) * (1 - Cow) * (1 - Dog) * (1 - Bird) * (1 - Cat) * (1 - Sheep)),
		#  'Arm': ['Person'],
		(Arm * (1 - Person)),
		#  'Muzzle': ['Horse', 'Cow', 'Dog', 'Sheep'],
		(Muzzle * (1 - Horse) * (1 - Cow) * (1 - Dog) * (1 - Sheep)),
		#  'Hoof': ['Horse'],
		(Hoof * (1 - Horse)),
		#  'Tail': ['Horse', 'Cow', 'Dog', 'Bird', 'Sheep', 'Cat', 'Aeroplane'],
		(Tail * (1 - Horse) * (1 - Cow) * (1 - Dog) * (1 - Bird) * (1 - Sheep) * (1 - Cat) * (1 - Aeroplane)),
		#  'Bottle_Body': ['Bottle'],
		(Bottle_Body * (1 - Bottle)),
		#  'Paw': ['Dog', 'Cat'],
		(Paw * (1 - Dog) * (1 - Cat)),
		#  'Aeroplane_Body': ['Aeroplane'],
		(Aeroplane_Body * (1 - Aeroplane)),
		#  'Wing': ['Aeroplane', 'Bird'],
		(Wing * (1 - Aeroplane) * (1 - Bird)),
		#  'Wheel': ['Aeroplane', 'Car', 'Bicycle', 'Bus', 'Motorbike'],
		(Wheel * (1 - Aeroplane) * (1 - Car) * (1 - Bicycle) * (1 - Bus) * (1 - Motorbike)),
		#  'Stern': ['Aeroplane'],
		(Stern * (1 - Aeroplane)),
		#  'Cap': ['Bottle'],
		(Cap * (1 - Bottle)),
		#  'Hand': ['Person'],
		(Hand * (1 - Person)),
		#  'Frontside': ['Car', 'Bus', 'Train'],
		(Frontside * (1 - Car) * (1 - Bus) * (1 - Train)),
		#  'Rightside': ['Car', 'Bus', 'Train'],
		(Rightside * (1 - Car) * (1 - Bus) * (1 - Train)),
		#  'Roofside': ['Car', 'Bus', 'Train'],
		(Roofside * (1 - Car) * (1 - Bus) * (1 - Train)),
		#  'Backside': ['Car', 'Bus', 'Train'],
		(Backside * (1 - Car) * (1 - Bus) * (1 - Train)),
		#  'Leftside': ['Car', 'Train', 'Bus'],
		(Leftside * (1 - Car) * (1 - Bus) * (1 - Train)),
		#  'Door': ['Car', 'Bus'],
		(Door * (1 - Car) * (1 - Bus)),
		#  'Mirror': ['Car', 'Bus'],
		(Mirror * (1 - Car) * (1 - Bus)),
		#  'Headlight': ['Car', 'Bus', 'Train', 'Motorbike', 'Bicycle'],
		(Headlight * (1 - Car) * (1 - Bus) * (1 - Train) * (1 - Motorbike) * (1 - Bicycle)),
		#  'Window': ['Car', 'Bus'],
		(Window * (1 - Car) * (1 - Bus)),
		#  'Plate': ['Car', 'Bus'],
		(Plate * (1 - Car) * (1 - Bus)),
		#  'Engine': ['Aeroplane'],
		(Engine * (1 - Aeroplane)),
		#  'Foot': ['Person', 'Bird'],
		(Foot * (1 - Person) * (1 - Bird)),
		#  'Chainwheel': ['Bicycle'],
		(Chainwheel * (1 - Bicycle)),
		#  'Saddle': ['Bicycle', 'Motorbike'],
		(Saddle * (1 - Bicycle) * (1 - Motorbike)),
		#  'Handlebar': ['Bicycle', 'Motorbike'],
		(Handlebar * (1 - Bicycle) * (1 - Motorbike)),
		#  'Train_Head': ['Train'],
		(Train_Head * (1 - Train)),
		#  'Beak': ['Bird'],
		(Beak * (1 - Bird)),
		#  'Pot': ['Pottedplant'],
		(Pot * (1 - Pottedplant)),
		#  'Plant': ['Pottedplant'],
		(Plant * (1 - Pottedplant)),
		#  'Horn': ['Cow', 'Sheep']
		(Horn * (1 - Cow) * (1 - Sheep)),

		# B: OBJECT --> [OBJECT-PARTS] RULES
		# Tvmonitor = ['Screen'],
		(Tvmonitor * (1 - Screen)),
		# Train = ['Coach', 'Leftside', 'Train_Head', 'Headlight', 'Frontside', 'Rightside', 'Backside', 'Roofside'],
		(Train * (1 - Coach) * (1 - Leftside) * (1 - Train_Head) * (1 - Headlight) * (1 - Frontside) * (1 - Rightside) * (1 - Backside) * (1 - Roofside)),
		# Person = ['Torso', 'Leg', 'Head', 'Ear', 'Eye', 'Ebrow', 'Mouth', 'Hair', 'Nose', 'Neck', 'Arm', 'Hand', 'Foot'],
		(Person * (1 - Torso) * (1 - Leg) * (1 - Head) * (1 - Ear) * (1 - Eye) * (1 - Ebrow) * (1 - Mouth) * (1 - Hair)* (1 - Nose) * (1 - Neck)* (1 - Arm) * (1 - Hand)* (1 - Foot)),
		# Boat = [],
		# Horse = ['Head', 'Ear', 'Muzzle', 'Torso', 'Neck', 'Leg', 'Hoof', 'Tail', 'Eye'],
		(Horse * (1 - Head) * (1 - Ear) * (1 - Muzzle) * (1 - Torso) * (1 - Neck) * (1 - Leg) * (1 - Hoof) * (1 - Tail) * (1 - Eye)),
		# Cow = ['Head', 'Ear', 'Eye', 'Muzzle', 'Torso', 'Neck', 'Leg', 'Tail', 'Horn'],
		(Cow * (1 - Head) * (1 - Ear) * (1 - Eye) * (1 - Muzzle) * (1 - Torso) * (1 - Neck) * (1 - Leg) * (1 - Tail) * (1 - Horn)),
		# Bottle = ['Bottle_Body', 'Cap'],
		(Bottle * (1 - Bottle_Body) * (1 - Cap)),
		# Dog = ['Head', 'Ear', 'Torso', 'Neck', 'Leg', 'Paw', 'Eye', 'Muzzle', 'Nose', 'Tail'],
		(Dog * (1 - Head) * (1 - Ear) * (1 - Torso)  * (1 - Neck) * (1 - Leg) * (1 - Paw) * (1 - Eye)* (1 - Muzzle)* (1 - Nose)* (1 - Tail)),
		# Aeroplane = ['Aeroplane_Body', 'Wing', 'Wheel', 'Stern', 'Engine', 'Tail'],
		(Aeroplane * (1 - Aeroplane_Body) * (1 - Wing) * (1 - Wheel)  * (1 - Stern) * (1 - Engine) * (1 - Tail)),
		# Car = ['Frontside', 'Rightside', 'Door', 'Mirror', 'Headlight', 'Wheel', 'Window', 'Plate', 'Roofside', 'Backside', 'Leftside'],
		(Car * (1 - Frontside) * (1 - Rightside) * (1 - Door)  * (1 - Mirror) * (1 - Headlight) * (1 - Wheel) * (1 - Window) * (1 - Plate) * (1 - Roofside) * (1 - Backside) * (1 - Leftside)),
		# Bus = ['Plate', 'Frontside', 'Rightside', 'Door', 'Mirror', 'Headlight', 'Window', 'Wheel', 'Leftside', 'Backside', 'Roofside'],
		(Bus * (1 - Plate)* (1 - Frontside) * (1 - Rightside) * (1 - Door) * (1 - Mirror) * (1 - Headlight) * (1 - Window) * (1 - Wheel) * (1 - Leftside) * (1 - Backside) * (1 - Roofside)),
		# Bicycle = ['Wheel', 'Chainwheel', 'Saddle', 'Handlebar', 'Headlight'],
		(Bicycle * (1 - Wheel) * (1 - Chainwheel) * (1 - Saddle) * (1 - Handlebar) * (1 - Headlight)),
		# Table = [],
		# Chair = [],
		# Bird = ['Head', 'Eye', 'Beak', 'Torso', 'Neck', 'Leg', 'Foot', 'Tail', 'Wing'],
		(Bird * (1 - Head) * (1 - Eye) * (1 - Beak) * (1 - Torso) * (1 - Neck) * (1 - Leg) * (1 - Foot) * (1 - Tail) * (1 - Wing)),
		# Cat = ['Head', 'Ear', 'Eye', 'Nose', 'Torso', 'Neck', 'Leg', 'Paw', 'Tail'],
		(Cat * (1 - Head) * (1 - Ear) * (1 - Eye) * (1 - Nose) * (1 - Torso) * (1 - Neck) * (1 - Leg) * (1 - Paw) * (1 - Tail)),
		# Motorbike = ['Wheel', 'Headlight', 'Handlebar', 'Saddle'],
		(Motorbike * (1 - Wheel) * (1 - Headlight) * (1 - Handlebar) * (1 - Saddle)),
		# Sheep = ['Head', 'Ear', 'Eye', 'Muzzle', 'Torso', 'Neck', 'Leg', 'Tail', 'Horn'],
		(Sheep * (1 - Head) * (1 - Ear) * (1 - Eye) * (1 - Muzzle) * (1 - Torso) * (1 - Neck) * (1 - Leg) * (1 - Tail) * (1 - Horn)),
		# Sofa = [],
		# Pottedplant = ['Pot', 'Plant']
		(Pottedplant * (1 - Pot) * (1 - Plant)),

		# C: OR ON THE OBJECTS
		((1 - Tvmonitor) * (1 - Train) * (1 - Person) * (1 - Boat) * (1 - Horse) * (1 - Cow) * (1 - Bottle) *
		 (1 - Dog) * (1 - Aeroplane) * (1 - Car) * (1 - Bus) * (1 - Bicycle) * (1 - Table) * (1 - Chair) *
		 (1 - Bird) * (1 - Cat) * (1 - Motorbike) * (1 - Sheep) * (1 - Sofa) * (1 - Pottedplant)),

		# D: OR ON THE PARTS
		# ((1 - Screen) *  (1 - Coach) *  (1 - Torso) * (1 - Leg) * (1 - Head) * (1 - Ear) * (1 - Eye) * (1 - Ebrow) *
		#  (1 - Mouth) * (1 - Hair) * (1 - Nose) * (1 - Neck) * (1 - Arm) * (1 - Muzzle) * (1 - Hoof) * (1 - Tail) *
		#  (1 - Bottle_Body) * (1 - Paw) * (1 - Aeroplane_Body) * (1 - Wing) * (1 - Wheel) * (1 - Stern) * (1 - Cap) *
		#  (1 - Hand) * (1 - Frontside) * (1 - Rightside) * (1 - Door) * (1 - Mirror) * (1 - Headlight) * (1 - Window) *
		#  (1 - Plate) * (1 - Roofside) * (1 - Backside) * (1 - Leftside) * (1 - Engine) * (1 - Foot) * (1 - Chainwheel) *
		#  (1 - Saddle) * (1 - Handlebar) * (1 - Train_Head) * (1 - Beak) * (1 - Pot) * (1 - Plant) * (1 - Horn))
	]

	if sum:
		losses = torch.sum(torch.stack(loss_fol_product_tnorm, dim=0), dim=1)
	else:
		losses = torch.stack(loss_fol_product_tnorm, dim=0)

	loss_sum = torch.squeeze(torch.sum(losses, dim=0))
	# print("Output", output)
	# print("Losses", losses)
	# print("Loss_sum", loss_sum)
	return loss_sum


if __name__ == "__main__":
	from pascalpart import name_list, name_ids
	for name in name_list:
		print(name)
	print(name_list, name_ids)
