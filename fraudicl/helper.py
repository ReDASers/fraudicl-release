import fraudicl.global_config as global_config
from dataclasses import dataclass
from time import time
from typing import Dict, Optional, Union
import pandas as pd
try:
    from IPython.display import Image, display
    IPYTHON_AVAILABLE = True
except ImportError:
    IPYTHON_AVAILABLE = False
import locale
import gc
import torch
from accelerate.utils import release_memory

@dataclass
class TimingEntry:
    """Container for timing information for a single section."""
    start: Optional[float] = None
    stop: Optional[float] = None
    elapsed: Optional[float] = None
    formatted: Optional[str] = None

class TimingManager:
    """Manager for timing code execution across different sections."""
    
    def __init__(self) -> None:
        """Initialize the timing manager with empty timing dictionary, each containing a TimingEntry."""
        self.timings: Dict[str, TimingEntry] = {}

    def start_time(self, section_name: str) -> None:
        """Start a timing sequence with given section name."""
        if section_name not in self.timings:
            self.timings[section_name] = TimingEntry()
        
        self.timings[section_name].start = time()

    def stop_time(self, section_name: str) -> None:
        """Stop a timing sequence, calculate elapsed time, and store formatted time."""
        entry = self.timings[section_name]
        entry.stop = time()
        if entry.start is not None:
            entry.elapsed = entry.stop - entry.start
            entry.formatted = self.format_time(entry.elapsed)
        else:
            raise ValueError(f"Section {section_name} has no start time found; first run start_time({section_name}).")
    
    def report_time(self, section_name: str) -> None:
        """Prints formatted elapsed time for the section."""
        if section_name not in self.timings or self.timings[section_name].formatted is None:
            raise ValueError(f"Section {section_name} has no elapsed time recorded; first run start_time({section_name}) and then stop_time({section_name}).")
        
        print(f"Total time for {section_name}: {self.timings[section_name].formatted}")
    
    def return_elapsed_time(self, section_name: str) -> float | None:
        """Returns the elapsed time."""
        return self.timings[section_name].elapsed
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert timings to a pandas DataFrame."""
        data = []
        for section, entry in self.timings.items():
            data.append({
                "section": section,
                "elapsed_seconds": entry.elapsed,
                "formatted_time": entry.formatted
            })
        return pd.DataFrame(data)

    @staticmethod
    def format_time(seconds: float) -> str:
        """Conditionally format total time for easier parsing."""
        if seconds >= 86400:
            final_time = seconds/86400
            unit = "day(s)"
        elif seconds >= 3600:
            final_time = seconds/3600
            unit = "hour(s)"
        elif seconds >= 60:
            final_time = seconds/60
            unit = "minute(s)"
        else:
            final_time = seconds
            unit = "second(s)"
            
        return f"{final_time:.2f} {unit}"

session_timer = TimingManager() # default instance

class WalkthroughManager:
    """Manager for displaying walkthrough images in sequence."""
    
    def __init__(self) -> None:
        """Initialize the walkthrough manager with its current step count."""
        self.step: int = 0

    def show_step(self, specific_step: Optional[Union[int, str]] = None) -> None:
        """Show the specific walkthrough image associated with the given step."""
        if not IPYTHON_AVAILABLE:
            raise RuntimeError("WalkthroughManager requires IPython (only available in Jupyter notebooks)")

        if specific_step is not None:
            display_step = specific_step
        else:
            self.step += 1
            display_step = self.step

        curr_img = f"{global_config.General.LOCAL_WALKTHROUGH_IMAGES}/step{display_step}.png"
        display(Image(filename=curr_img))
    
    def set_step(self, specific_step: int) -> None:
        """Set the current step to the specified amount (can use this to reset to 0)."""
        if specific_step < 0:
            self.step = 0
        else:
            self.step = specific_step

    def reset(self) -> None:
        """Reset the step counter to 0."""
        self.step = 0

    def get_current_step(self) -> int:
        """Get the current step number."""
        return self.step

    def next_step(self) -> None:
        """Advance to and display the next step."""
        self.show_step()

    def previous_step(self) -> None:
        """Go back to and display the previous step."""
        if self.step > 1:
            self.step -= 2  # Subtract 2 because show_step() will add 1
            self.show_step()
        else:
            self.show_step(1)

def setup_locale() -> None:
    """Monkey patching locale encoding to UTF-8 (helps standardize text encodings across different systems)."""
    def getpreferredencoding(do_setlocale: bool = True) -> str:
        return "UTF-8"
    
    locale.getpreferredencoding = getpreferredencoding
    print("Locale encoding set to UTF-8")

def memclear(free_debug_stack: bool = False) -> None:
    """Clear GPU and CPU memory to prevent OOM errors during batch processing."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    release_memory()
    gc.collect()
    
    # Clear debug stack
    if free_debug_stack:
        print("Clearing debug stack traces...")
        try:
            1/0  # Intentional division by zero to clear exception traces
        except ZeroDivisionError:
            pass

def determine_device(option: str) -> str:
    """Determine the best device for embedding computation; needed for Haystack components."""
    if option == "auto":
        if torch.cuda.is_available():
            return "cuda:0"
        else:
            return "cpu"
    else:
        return option
    
def legacy_format_to_fro_tag(input: Union[str, int, bool]) -> Union[str, bool]:
    """Convert legacy format to and from tag."""
    if isinstance(input, bool):
        result = "old" if input else "new"
    elif isinstance(input, int):
        result = "old" if input==1 else "new"
    else:
        result = True if input.lower() == "old" else False
    return result