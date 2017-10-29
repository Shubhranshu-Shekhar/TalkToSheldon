import json
from collections import namedtuple


Utterance = namedtuple("Utterance", field_names=["others_utterance", "character_utterance"])


class BBTDatasetReader(object):
    def __init__(self, file_path):
        self.file_path = file_path

    def read(self, character_name="Sheldon"):
        utterances = []
        with open(self.file_path, "r") as data_fp:
            try:
                data_json = json.loads(data_fp.read())
                for episode_text in data_json.values():
                    ep_utterances = BBTDatasetReader.process_episode(episode_text=episode_text,
                                                                     character_name=character_name)
                    utterances.extend(ep_utterances)

            except Exception as e:
                print "Error reading json"
                raise e
        return utterances

    @staticmethod
    def process_utterance(utterance_text):
        pos = utterance_text.find(":")
        if pos == -1: return None
        character = utterance_text[:pos]
        if character.lower() in ["scene"]: return None
        return character, utterance_text[pos+1:]

    @staticmethod
    def process_episode(episode_text, character_name="Sheldon"):
        episode_text = episode_text.encode("ascii", errors="ignore")
        episode_text = episode_text.replace("\t", " ")
        episode_utterances = episode_text.split("\n")
        episode_chunks = []
        curr_chunk_other, curr_chunk_char = "", ""

        for utterance in episode_utterances:
            utterance = BBTDatasetReader.process_utterance(utterance)
            if not utterance:
                continue

            u_char, u_text = utterance
            if u_char != character_name:
                curr_chunk_other += u_text + "."
            else:
                curr_chunk_char = u_text
                curr_chunk_char = curr_chunk_char.strip()
                curr_chunk_other = curr_chunk_other.strip()
                episode_chunks.append(Utterance(
                    others_utterance=curr_chunk_other, character_utterance=curr_chunk_char))
                curr_chunk_other = ""
        return episode_chunks