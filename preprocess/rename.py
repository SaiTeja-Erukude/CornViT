import os
from   tqdm     import tqdm


def rename_files(input_dir: str, word: str, new_word: str = '') -> bool:
    """
    Desc: 
        This method renames the files that has {word} in the filename
    Args:
        input_dir (str): Path to the input directory 
        word (str): A word to look for in the filename
        new_word (str): A new word that is used to replace
    Returns:
        True, if renaming operation is success, else False.
    """
    try:        
        if not os.path.isdir(input_dir):
            print(f"The directory {input_dir} does not exist.")
            return False
        
        for file in tqdm(os.listdir(input_dir)):
            basename, ext = os.path.splitext(file)

            if word in basename:
                new_name = f"{basename.replace(word, new_word)}{ext}"
                
                # Construct full paths for renaming
                old_file_path = os.path.join(input_dir, file)
                new_file_path = os.path.join(input_dir, new_name)
                
                try:
                    # Rename the file
                    os.rename(old_file_path, new_file_path)
                except:
                    continue
        
        return True
    
    except Exception as rename_ex:
        print(f"An error occurred while renaming: {rename_ex}")
        return False

   


if __name__ == "__main__":
    inp_dir = ""
    word = ""
    new_word = ""
    rename_files(inp_dir, word, new_word)
    