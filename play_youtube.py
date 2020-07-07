import pafy, vlc

url = "https://www.youtube.com/watch?v=hT_nvWreIhg"
instance = vlc.Instance()
media = instance.media_new(url)
media.get_mrl()
media_list = instance.media_list_new([url])  # A list of one movie

player = instance.media_player_new()
player.set_media(media)
player.play()