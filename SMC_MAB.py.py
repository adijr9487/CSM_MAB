import numpy as np
from scipy.stats import bernoulli

# Show array


def show(arr):
    for ele in arr:
        print(ele)


def getUserChannel(user_idx, channel_mapping):
    return list(channel_mapping.keys())[list(channel_mapping.values()).index([user_idx])]


def checkCollision(cur_sampling_channel):
    my_set = set(cur_sampling_channel)
    # channel_map = [[]]
    channel_dict = {}

    for k in my_set:
        channel_dict[k] = []

    for i in range(len(cur_sampling_channel)):
        if(cur_sampling_channel[i] in my_set):
            channel_dict[cur_sampling_channel[i]].append(i)

    return channel_dict


def CFL(num_of_channels, num_of_user, channel_reward, channel_selection_count):
    des_param = 0.1
    channel_selection_prob = [
        [(1/num_of_channels) for _ in range(num_of_channels)] for _ in range(num_of_user)]
    flag = False

    while(flag == False):

        # user transmission to channel
        cur_sampling_channel = [np.random.choice(
            a=int(num_of_channels), p=channel_selection_prob[i]) for i in range(num_of_user)]

        # channel selection increment for their sampling
        for i in range(len(cur_sampling_channel)):
            channel_selection_count[i][cur_sampling_channel[i]] += 1

        # channel -> user
        channel_mapping = checkCollision(
            cur_sampling_channel)  # check collision

        for channel in channel_mapping:
            if(len(channel_mapping[channel]) > 1):
                for user in channel_mapping[channel]:
                    # storing curr prob state for reuse
                    prev = channel_selection_prob[user][channel]

                    # updating non-collision channel prob
                    channel_selection_prob[user] = [((1-des_param)*channel_selection_prob[user][j]) + (
                        des_param/(num_of_channels-1)) for j in range(num_of_channels)]

                    # updating collided channel prob
                    channel_selection_prob[user][channel] = (1-des_param)*prev

                    # setting reward of collided channel
                    channel_reward[user][channel] += bernoulli.rvs(
                        channel_selection_prob[user][channel])

            else:
                # make probabilty 1 and rest other as 0
                for user in channel_mapping[channel]:
                    channel_selection_prob[user] = [
                        0 for _ in range(num_of_channels)]

                    channel_selection_prob[user][channel] = 1

                    # set the reward of selected channel
                    # Bernoulli.rvs(n%) will return 1 for n amount of time.
                    channel_reward[user][channel] += bernoulli.rvs(
                        channel_selection_prob[user][channel])

        # checking if the configuration is orthogonalised
        if len(channel_mapping) == num_of_user:
            # print("orthogonal configuration", channel_mapping)
            flag = True

    # print("prob", channel_selection_prob)
    # print("reward", channel_reward)
    # print("selection_count", channel_selection_count)

    return channel_mapping


def rankChannels(channel_reward, channel_selection_count, num_of_user, num_of_channels, time):

    arr = []
    reverse_arr = []
    for user in range(num_of_user):
        list_arr = []
        for channel in range(num_of_channels):
            if channel_selection_count[user][channel] == 0:
                list_arr.append([0, channel])
            else:
                Ik = (channel_reward[user][channel] /
                      channel_selection_count[user][channel])
                Ik += np.sqrt((2*np.log(time)/np.exp(1)) /
                              channel_selection_count[user][channel])
                list_arr.append([Ik, channel])

        arr.append(list_arr)
        list_arr.sort(reverse=True)
        reverse_arr.append(list_arr)

    return arr, reverse_arr


def chooseInitiator(channel_rank, num_of_user, user_mapping):
    epsi = 0.2
    raise_flag = []

    # All users raising flags
    # print(list(channel_mapping.keys())[list(channel_mapping.values()).index([0])])
    for user_idx in range(num_of_user):

        # user preference
        # != user channel sampling
        if channel_rank[user_idx][0][1] != user_mapping[user_idx]:
            if len(channel_rank[user_idx]) != 0:  # if they are participating
                # raise flag with EPSILON probability
                raise_flag.append(bernoulli.rvs(epsi))
                # print(channel_rank[user_idx], user_mapping[user_idx])
            else:  # if they don't want to participate
                raise_flag.append(0)
        else:
            raise_flag.append(0)

    if sum(raise_flag) == 1:
        return raise_flag.index(1)
    else:
        return -1


def transmit_and_learn(user_idx, user_mapping, channel_reward, channel_selection_count):
    channel_idx = user_mapping[user_idx]

    # channel selection increment for their sampling
    channel_selection_count[user_idx][channel_idx] += 1

    channel_reward[user_idx][channel_idx] += 1


def main():
    num_of_channels = 10
    num_of_user = 6
    pref = -1
    initiator = -1
    channel_rank = None
    responder = -1
    num_of_swap = 0

    channel_reward = [[0 for i in range(num_of_channels)]
                      for j in range(num_of_user)]

    channel_selection_count = [
        [(0) for i in range(num_of_channels)] for j in range(num_of_user)]

    # CFL for all users
    channel_mapping = CFL(num_of_channels, num_of_user,
                          channel_reward, channel_selection_count)
    user_mapping = {v[0]: k for k, v in channel_mapping.items()}
    # print("Channel Reward")
    # show(channel_reward)

    # print("\n\n")

    # print("Sample Count")
    # show(channel_selection_count)

    # print("\n\n")

    # print('Channel Configuration')
    # print(channel_mapping)

    for frame in range(1, 450):
        
        # Beginning of Super Frame ( INITIALIZTION FRAME )
        if frame % (2*num_of_channels) == 1:

            ucb_indexing, channel_rank = rankChannels(
                channel_reward, channel_selection_count, num_of_user, num_of_channels, frame)
            # channel_rank is channel preferences for each user
            # print("channel ranking for each user at beginning of frame", channel_rank)
            initiator = chooseInitiator(
                channel_rank, num_of_user, user_mapping)
            pref = 0

        else:  # Upcomming MINI-frames of SUPER FRAME
            if initiator > -1 and pref < len(channel_rank[initiator]):

                initiator_choice = channel_rank[initiator][pref]
                initiator_ucb_Ik = initiator_choice[0]  # preference channel IK
                # preference channel number
                initiator_channel_idx = initiator_choice[1]

                # print("initiator: ", [initiator])
                # print("pref channel: ", initiator_channel_idx)

                if initiator_channel_idx in channel_mapping:
                    responder = channel_mapping[initiator_channel_idx][0]

                    responder_ucb_Ik = ucb_indexing[responder][initiator_channel_idx][0]

                    if responder_ucb_Ik <= initiator_ucb_Ik:  # swap
                        # initiator transmit to channel
                        user_mapping[responder] = user_mapping[initiator]
                        user_mapping[initiator] = initiator_channel_idx
                        channel_mapping = {v: [k]
                                           for k, v in user_mapping.items()}
                        num_of_swap += 1
                        pref = len(channel_rank[initiator])  # stop upgradation
                        print("Channel Reward", channel_reward[3])
                        print("Current Channel", user_mapping[3])

                    else:  # can not swap
                        pref += 1  # jump over next preference
                else:
                    user_mapping[initiator] = initiator_channel_idx
                    channel_mapping = {v: [k] for k, v in user_mapping.items()}
                    num_of_swap += 1
                    pref = len(channel_rank[initiator])  # stop upgradation
                    # print("After M-F: ", channel_mapping)
                    # print(user_mapping)
                    print("Channel Reward", channel_reward[3])
                    print("Current Channel", user_mapping[3])

            else:
                # transmit and learn
                for i in range(num_of_user):
                    if i != responder and i != initiator:  # if not responder and initiator
                        # they will transmit
                        transmit_and_learn(
                            i, user_mapping, channel_reward, channel_selection_count)
                        user_mapping = {v[0]: k for k,
                                        v in channel_mapping.items()}

    # print("Channel Reward")
    # show(channel_reward)

    # print("\n\n")

    # print("Sample Count")
    # show(channel_selection_count)

    # print("\n\n")

    # print('Channel Configuration')
    # print(channel_mapping)

    # print("\n\n Channel rank")
    # show(channel_rank)

    # print("num_of_swap", num_of_swap)


if __name__ == "__main__":
    main()
