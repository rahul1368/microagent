# Recipe for the README gif.
#
# This is a vhs tape (https://github.com/charmbracelet/vhs). It records the
# demo below into assets/demo.gif with no manual screen capture. The recipe is
# text, so anyone can re-render the gif and it stays in sync with the code.
#
# Render it with:
#   brew install vhs        # also pulls ttyd and ffmpeg
#   vhs demo.tape

Output assets/demo.gif

Set FontSize 20
Set Width 1100
Set Height 520
Set Padding 24
Set Theme "Catppuccin Mocha"

Type "python3 demo.py"
Sleep 600ms
Enter
Sleep 3s
