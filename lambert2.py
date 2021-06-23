import torch


def lambertw(z, step_tol=1e-8):
	"""
	Calculate the k=0 branch of lambert w function.

	Inverse of z*exp(z), i.e. w(z)*exp(w(z)) = z

	Iterative algorithm from
	https://www.quora.com/How-is-the-Lambert-W-Function-computed
	"""
	with torch.no_grad():
		i = 0
		w = torch.log(1 + z)
		step = w
		while torch.max(torch.abs(step)) > step_tol and i < 20:
			ew = torch.exp(w)
			numer = w*ew - z
			step = numer/(ew*(w+1) - (w+2)*numer/(2*w + 2))
			w = w - step
			i+=1

	return w


def lambertw_grad(z, w=None):
	"""From implicit differentiation."""
	if w is None:
		w = lambertw(z)
	return w / (z*(1 + w))