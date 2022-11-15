import torch
import torch.nn.functional as f

class DomainAttentionLayer(torch.nn.Module):
	def __init__(self, embedding_dim):
		super(DomainAttentionLayer, self).__init__()
		self.embedding_dim = embedding_dim
		self.keys = torch.nn.Linear(embedding_dim, embedding_dim)
		self.queries = torch.nn.Linear(embedding_dim, embedding_dim)
		self.values = torch.nn.Linear(embedding_dim, embedding_dim)

	def forward(self, x, domain_x):
		keys_x = self.keys(domain_x)
		queries_x = self.queries(x)
		values_x = self.values(domain_x)
		scores = torch.mm(queries_x, keys_x.transpose(0, 1))
		scale = keys_x.size(-1) ** 0.5
		softmax = f.softmax(scores / scale, dim=-1)
		embeds = torch.mm(softmax, values_x)
		return embeds