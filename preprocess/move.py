import os
import random
import shutil
from   tqdm         import tqdm


######################
#
######################
def move_n_random_files(dir1: str, dir2: str, n: int) -> bool: 
    """
    Desc: 
        This method moves 'n' random files from the source directory (dir1) to destination directory (dir2).
    Args:
        dir1 (str): Path to the source directory.
        dir2 (str): Path to the destination directory.
        n (int): The number of random files to be moved.
    Returns:
        True, if the operation was successful, otherwise False.
    """
    try:
        # Check if the source and destination directory exists
        if not os.path.isdir(dir1):
            print(f"Directory '{dir1}' does not exist.")
            return False
        
        if not os.path.isdir(dir2):
            print(f"Directory '{dir2}' does not exist. Creating it.")
            os.makedirs(dir2)
        
        # Get all files (not directories) in the specified directory
        all_files = [f for f in os.listdir(dir1) if os.path.isfile(os.path.join(dir1, f))]
        
        if len(all_files) < n:
            print(f"Cannot move '{n}' files, directory only contains '{len(all_files)}' files.")
            return False
        
        files_to_move = random.sample(all_files, n)
        
        for file in tqdm(files_to_move):
            source_path      = os.path.join(dir1, file)
            destination_path = os.path.join(dir2, file)

            # Move the file to the destination directory
            shutil.move(source_path, destination_path)
        
        return True
    
    except Exception as move_ex:
        print(f"Error occurred while moving files: {str(move_ex)}.")
        return False


def copy_n_random_files(dir1: str, dir2: str, n: int) -> bool:
    """
    Desc:
        Randomly select and copy 'n' files from one directory to another.
    Args:
        dir1 (str): Path to the source directory.
        dir2 (str): Path to the destination directory. Will be created if it doesn't exist.
        n (int): Number of files to randomly copy.
    Returns:
        bool: True if the operation was successful, False otherwise.
    """
    try:
        # Check if the source directory exists
        if not os.path.isdir(dir1):
            print(f"Directory '{dir1}' does not exist.")
            return False

        # Ensure destination directory exists
        if not os.path.isdir(dir2):
            print(f"Directory '{dir2}' does not exist. Creating it.")
            os.makedirs(dir2)

        # Get list of all files (not directories) in source
        all_files = [f for f in os.listdir(dir1) if os.path.isfile(os.path.join(dir1, f))]

        if len(all_files) < n:
            n = len(all_files)

        print(f"Copying '{n}' files to '{dir2}'...")
        files_to_copy = random.sample(all_files, n)

        for file in tqdm(files_to_copy, desc="Copying files"):
            source_path = os.path.join(dir1, file)
            destination_path = os.path.join(dir2, file)
            shutil.copy(source_path, destination_path)

        return True

    except Exception as copy_ex:
        print(f"Error occurred while copying files: {str(copy_ex)}.")
        return False
    

######################
#
######################
def copy_n_unique_files(dir1, dir2, output_dir, n):
    """
    Desc: 
        This method iterates files in dir1 and checks if they are not present in dir2. If not present, copies the file to output_dir. Moves 'n' files in total.
    Args:
        dir1 (str): Path to directory 1.
        dir2 (str): Path to directory 2.
        output_dir (str): Path to the destination directory.
        n (int): The number of random files to be moved.
    Returns:
        True, if the operation was successful, otherwise False.
    """
    try:
        # List all files in dir1 and dir2
        dir1_files = [f for f in os.listdir(dir1) if os.path.isfile(os.path.join(dir1, f))]
        dir2_files = [f for f in os.listdir(dir2) if os.path.isfile(os.path.join(dir2, f))]

        # Filter out files that already exist in dir2
        unique_files = [f for f in dir1_files if f not in dir2_files]

        # If no unique files are found
        if not unique_files:
            print("No unique files to move.")
            return False

        # Randomly select 'n' files to copy (make sure we don't select more than we have)
        files_to_copy = random.sample(unique_files, min(n, len(unique_files)))

        # Copy selected files to output_dir
        files_copied = 0
        for file in files_to_copy:
            src_path  = os.path.join(dir1, file)
            dest_path = os.path.join(output_dir, file)
            shutil.copy(src_path, dest_path)
            files_copied += 1
            print(f"Copied: {file}")

        print(f"Total files copied: {files_copied}")
        return True
    
    except Exception as copy_ex:
        print(f"An error occurred while copying: {copy_ex}")
        return False
 

######################
#
######################
if __name__ == "__main__":
    dir1 = ""
    dir2 = ""
    n    = 0
    
    move_n_random_files(dir1, dir2, n)