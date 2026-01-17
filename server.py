from openreward.environments import Server

from r2e_gym import R2EGym

if __name__ == "__main__":
    server = Server([R2EGym])
    server.run()
