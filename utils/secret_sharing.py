import torch
class Secret(object):
    def __init__(self, n_agents, secret = None, Q=2**31 - 1, precision=4, base=10):
        self.secret = secret
        self.Q = Q
        self.precision = precision
        self.base = base
        self.n_agents = n_agents
        self.shares = self.encrypt(secret) if secret is not None else []

    def reveal(self):
        return self.decrypt(self.shares)
    
    def __repr__(self):
        return "Secret(%d)" % self.reveal()
    
    def __add__(x, y):
        z = Secret()
        z.shares = x.shares + y.shares
        return z
        
    def __sum__(x):
        z = Secret()
        z.shares = sum([item.shares for item in x])
        return z
    
    def __sub__(x, y):
        z = Secret()
        z.shares = x.shares - y.shares
        return z
    
    def __mul__(x, y):
        print("warning: multiplication is not secure")
        return x.reveal() * y.reveal()
    
    def get_i_th_share(self, i):
        return self.shares[i]
    
    
    def encoder(self, x):
        encoding = (self.base**self.precision * x % self.Q).clone().detach().int()
        return encoding

    # Fixed point decoding
    def decoder(self, x):
        x[x > self.Q/2] = x[x > self.Q/2] - self.Q
        return x / self.base**self.precision
    
    # Additive secret sharing
    def encrypt(self, x):
        shares = []
        for i in range(self.n_agents-1): 
            shares.append(torch.randint(0, self.Q, x.shape, device=self.device))
        shares.append(self.Q - sum(shares) % self.Q + self.encoder(x))
        shares = torch.stack(shares, dim=0)
        return shares
    
    def decrypt(self, shares):
        return self.decoder(torch.sum(shares, dim=0) % self.Q)