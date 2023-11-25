import pandas as pd
import numpy as np
from sklearn import model_selection, metrics, preprocessing
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
df = pd.read_csv("D:\\HUTECH\\NCKH\\HeThongDeXuatMuaSamThongMinh\\documentary\\ratings.csv")
df.info()
df.userId.nunique()
df.movieId.nunique()
df.rating.value_counts()



# Tranning dataset Wrapper
class MovieDataset:
    def __init__(self, users, movies, ratings):
        self.users = users
        self.movies = movies
        self.ratings = ratings

    def __len__(self):
        return len(self.users)

    def __getitem__(self, item):
        users = self.users(item)
        movies = self.movies(item)
        ratings = self.ratings(item)

        return {
            "users": torch.tensor(users, dtype=torch.long),
            "movies": torch.tensor(movies, dtype=torch.long),
            "ratings": torch.tensor(ratings, dtype=torch.long),
        }



#create the model
class RecSysModel(nn.Module):
    def __init__(self, n_users, n_movies):
        super().__init__()
        self.user_embed = nn.Embedding(n_users, 32)
        self.movie_embed = nn.Embedding(n_movies, 32)
        self.out = nn.Linear(64, 1)

    def forvard(self, users, movies, ratings = None):
        users_embeds = self.user_embed(users)
        movies_embeds = self.movie_embed(movies)
        output = torch.cat([users_embeds, movies_embeds], dim=1)
        output = self.out(output)
        return output



#encode the user and movie id to start from 0 so we don't run into index out of bound with embedding
lbl_user = preprocessing.LabelEncoder()
lbl_movie = preprocessing.LabelEncoder()
df.userId = lbl_user.fit_transform(df.userId.values)
df.movieId= lbl_movie.fit_transform(df.movieId.values)

df_train, df_valid = model_selection.train_test_split(
    df, test_size=0.1, random_state=42, stratify=df.rating.values
)

train_dataset = MovieDataset(
    users= df_train.userId.values,
    movies=df_train.movieId.values,
    ratings=df_valid.rating.values
)
valid_dataset = MovieDataset(
    users=df_train.userId.values,
    movies=df_valid.movieId.values,
    ratings=df_valid.rating.values
)



train_loader = DataLoader(dataset=train_dataset,
                          batch_size=4,
                          shuffle=True,
                          num_workers=2)
validation_loader = DataLoader(dataset=train_dataset,
                               batch_size=4,
                               shuffle=True,
                               num_workers=2)
dataiter = iter(train_loader)
dataloader_data = dataiter.next()
print(dataloader_data)



model =  RecSysModel(
    n_users = len(lbl_user.classes_),
    n_movies= len(lbl_movie.classes_),
).to(device)

optimizer = torch.optim.Adam(model.parameters())
sch = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.7)

loss_func = nn.MSELoss()

print(len(lbl_user.classes_))
print(len(lbl_movie.classes_))
print(df.movieId.max())
print(len(train_dataset))


#Manually run a forward path
print(dataloader_data['user'])

print(dataloader_data['user'].size())
print(dataloader_data['movies'] )
print(dataloader_data['movies'].size())

user_emabed = nn.Embedding(len(lbl_user.classes_), 32)
movie_embed = nn.Embedding(len(lbl_movie.classes_), 32)
out = nn.Linear(64,1)



user_emabed = user_emabed(dataloader_data['users'])
movie_embed = movie_embed(dataloader_data['movies'])
print(f"user_emabeds {user_emabed.size()}")
print(f"user_emabeds {user_emabed}")
print(f"movie_emabeds {movie_embed.size()}")
print(f"user_emabeds {movie_embed}")



output = torch.cat([user_emabed, movie_embed, movie_embed], dim =1)
print(f"output: {output.size()}")
print(f"output: {output}")
output = out(output)
print(f"output: {output}")



with torch.no_grad():
    model_output = model(dataloader_data['user'], dataloader_data["movies"])
    print(f"model_out: {model_output}, size: {model_output.size()}")


rating = dataloader_data["rating"]
print(rating)
print(rating.view(4,-1))
print(model_output)

print(rating.sum())

print(model_output.sum() - rating.sum())


#run the traning loop
epochs = 1;
total_loss = 0
plot_steps, print_steps = 5000, 5000
step_cnt = 0
all_losses_list = []

model.train()
for epoch_i in range (epochs):
        for i, train_data in enumerate(train_loader):
            output= model(train_data["user"],
                          train_data["movies"])
        rating = train_data["ratings"].view(4,-1).to(torch.float32)

        loss = loss_func(output, rating)
        total_loss = total_loss + loss.sum().item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        step_cnt = step_cnt + len(train_data["users"])

        if(step_cnt % plot_steps == 0 ):
            avg_loss = total_loss/ (len(train_data["users"]) * plot_steps)
            print(f"epoch {epoch_i} loss at step: {step_cnt} is {avg_loss}")
            total_loss = 0



plt.figure()
plt.plot(all_losses_list)
plt.show()



#Evaluation with RMSE
from sklearn.metrics import mean_squared_error

model_output_list = []
target_rating_list = []

model.eval()

with torch.no_grad():
    for i, batched_data in enumerate(validation_loader):
        model_output = model(batched_data['users'],
                             batched_data['movies'])
        model_output_list.append(model_output.sum().item() / len(batched_data['users']) )
        target_rating = batched_data["ratings"]
        target_rating_list.append(target_rating.sum().item() / len(batched_data['users']))
        print(f"model_output: {model_output}, target_rating: {target_rating}")

#squared If True returns MSE value, if False return RMSE value.
rms = mean_squared_error(target_rating_list, model_output_list, squared=False)
print(f"rms: {rms}")




#Evaluation with Recall@K
from collections import defaultdict

user_est_true = defaultdict(list)

with torch.no_grad():
    for i, batched_data  in enumerate(validation_loader):
        users = batched_data['users']
        movies = batched_data['movies']
        rating = batched_data['ratings']

        model_output = model(batched_data['users'], batched_data['movies'])

        for i in range(len(users)):
            user_id = users[i].item()
            movie_id = movies[i].item()
            pred_rating = model_output[i][0].item()
            true_rating = rating[i].item()

            print(f"{user_id}, {movie_id}, {pred_rating}, {true_rating}")
            user_est_true[user_id].append((pred_rating, true_rating))



with torch.no_grad():
    precisions = dict()
    recalls = dict()

    k = 100
    threshold = 3.5

    for uid, user_ratings in user_est_true.items():
        user_ratings.sort(key=lambda x:x[0], reverse=True)
        n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)
        n_rel_k = sum((est >= threshold) for (est, _) in user_ratings[:k])
        n_rel_and_rec_k = sum(
            ((true_r >= threshold) and(est >= threshold))
            for (est, true_r) in user_ratings[:k]
        )
        print(f"uid {uid}, n_rel{n_rel}, n_rec_k {n_rel_k}, n_rel_and_rec_k {n_rel_and_rec_k} ")

        precisions[uid] = n_rel_and_rec_k / n_rel_k if n_rel_k != 0 else 0

        recalls[uid] = n_rel_and_rec_k / n_rel if n_rel != 0 else 0



print(f"precision @ {k}: {sum(prec for prec in precisions.values()) / len(precisions)}")
print(f"recall @ {k} : {sum(rec for rec in recalls.values()) / len(recalls)}")