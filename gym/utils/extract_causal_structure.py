import argparse
import pickle


def extract_causal_structure(old_dataset_path, new_dataset_path):
    with open(old_dataset_path, 'rb') as file:
        trajectories = pickle.load(file)

    for tr in trajectories:
        observations = tr["observations"]
        tr["causal_structure"] = observations[:, -6:]
        tr["observations"] = observations[:, :-6]

    with open(new_dataset_path, 'wb') as file:
        pickle.dump(trajectories, file, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--old_dataset_path', type=str)
    parser.add_argument('--new_dataset_path', type=str)

    args = parser.parse_args()
    old_dataset_path, new_dataset_path = args.old_dataset_path, args.new_dataset_path
    extract_causal_structure(old_dataset_path, new_dataset_path)
