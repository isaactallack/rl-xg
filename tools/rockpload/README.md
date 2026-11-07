# Rockpload (optional)

[Rockpload](https://github.com/LEX0RE/rockpload) signs in to your Epic account, fetches replays from Rocket League's match history, and uploads them to websites. The dashboard uses it to capture games that Rocket League didn't save to the `Demos` folder.

It isn't bundled in this repo. To use it:

1. Download the Windows build (tested with v1.6.0) from the [Rockpload releases](https://github.com/LEX0RE/rockpload/releases).
2. Put the executable in this folder, with a name starting `rockpload` and ending `.exe` (for example `tools/rockpload/rockpload.exe`).
3. Start the dashboard. It adds an enabled **RL xG dashboard** upload target to Rockpload's settings (Ranked Doubles only) and launches it. Sign in with Epic in Rockpload's window the first time.

Rockpload uploads when you close Rocket League and about every 45 minutes. To get games during play, turn off *No Upload if player is online* in its behaviour settings.
