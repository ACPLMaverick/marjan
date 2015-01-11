using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.ComponentModel;
using System.IO;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Input;
using System.Xml.Serialization;
using XMLApp.Model;

namespace XMLApp.ViewModel
{
    class MainWindowViewModel : INotifyPropertyChanged
    {
        public event PropertyChangedEventHandler PropertyChanged;
        public String XmlPath { get; private set; }
        public String SchemaPath { get; private set; }

        //Collections
        public Płytoteka XmlFile { get; set; }

        private ObservableCollection<Gatunek> _genres;
        public ObservableCollection<Gatunek> Genres
        {
            get { return _genres; }
            set
            {
                if (_genres != value) _genres = value;
                OnPropertyChanged();
            }
        }

        private ObservableCollection<Płyta> _cds;
        public ObservableCollection<Płyta> Cds
        {
            get { return _cds; }
            set
            {
                if (_cds != value) _cds = value;
                OnPropertyChanged();
            }
        }

        private Płyta _cd;
        public Płyta Cd
        {
            get { return _cd; }
            set
            {
                if (_cd != value) _cd = value;
                OnPropertyChanged();
            }
        }

        private Gatunek _genre;
        public Gatunek Genre
        {
            get { return _genre; }
            set
            {
                if (_genre != value) _genre = value;
                OnPropertyChanged();
            }
        }

        private Utwór _song;
        public Utwór Song
        {
            get { return _song; }
            set
            {
                if (_song != value) _song = value;
                OnPropertyChanged();
            }
        }

        private String _cName;
        public String CommandName
        {
            get { return _cName; }
            set
            {
                if (_cName != value) _cName = value;
                OnPropertyChanged();
            }
        }

        private Visibility _commandVisibility;
        public Visibility CommandVisibility
        {
            get { return _commandVisibility; }
            set
            {
                if (_commandVisibility != value) _commandVisibility = value;
                OnPropertyChanged();
            }
        }

        private Visibility _editAlbumVisibility;
        public Visibility EditAlbumVisibility
        {
            get { return _editAlbumVisibility; }
            set
            {
                if (_editAlbumVisibility != value) _editAlbumVisibility = value;
                OnPropertyChanged();
            }
        }

        private Visibility _editSongVisibility;
        public Visibility EditSongVisibility
        {
            get { return _editSongVisibility; }
            set
            {
                if (_editSongVisibility != value) _editSongVisibility = value;
                OnPropertyChanged();
            }
        }

        public Commands OpenFileCommand { get; set; }
        public Commands SaveFileCommand { get; set; }
        public Commands ShowSongsCommand { get; set; }
        public Commands AddAlbumCommand { get; set; }
        public Commands RemoveAlbumCommand { get; set; }
        public Commands EditAlbumCommand { get; set; }
        public Commands AddSongCommand { get; set; }
        public Commands RemoveSongCommand { get; set; }
        public Commands EditSongCommand { get; set; }

        public MainWindowViewModel()
        {
            this.XmlPath = "lab2.xml";
            this.SchemaPath = "zewnetrznySchema.xsd";
            this.CommandName = "Pokaż utwory";
            this.CommandVisibility = Visibility.Hidden;
            this.EditAlbumVisibility = Visibility.Hidden;
            this.EditSongVisibility = Visibility.Hidden;
            this.OpenFileCommand = new Commands(OpenFileAction, CanExecuteAlwaysTrue);
            this.SaveFileCommand = new Commands(SaveFileAction, CanExecuteAlwaysTrue);
            this.ShowSongsCommand = new Commands(ShowSongsAction, CanExecuteAlwaysTrue);
            this.AddAlbumCommand = new Commands(AddAlbumAction, CanExecuteAlwaysTrue);
            this.RemoveAlbumCommand = new Commands(RemoveAlbumAction, CanExecuteAlwaysTrue);
            this.EditAlbumCommand = new Commands(EditAlbumAction, CanExecuteAlwaysTrue);
            this.AddSongCommand = new Commands(AddSongAction, CanExecuteAlwaysTrue);
            this.RemoveSongCommand = new Commands(RemoveSongAction, CanExecuteAlwaysTrue);
            this.EditSongCommand = new Commands(EditSongAction, CanExecuteAlwaysTrue);
        }

        private void OpenFileAction(object obj)
        {
            XmlSerializer serializer = new XmlSerializer(typeof(Płytoteka));
            using (FileStream file = new FileStream(this.XmlPath, FileMode.Open))
            {
                this.XmlFile = new Płytoteka();
                Płytoteka tmp = (Płytoteka)serializer.Deserialize(file);
                this.XmlFile = tmp;
            }
            this.Cds = this.XmlFile.płyty.płyty;
            this.Genres = this.XmlFile.gatunki.genres;
            GenreSetters();
        }

        private void SaveFileAction(object obj)
        {
            if(this.Cds != null)
            {
                RefreshGenre();
                XmlSerializer serializer = new XmlSerializer(typeof(Płytoteka));
                using (FileStream file = new FileStream(this.XmlPath, FileMode.Create))
                {
                    serializer.Serialize(file, this.XmlFile);
                }
            }
        }

        private void ShowSongsAction(object obj)
        {
            if(this.Cds != null)
            {
                if (this.CommandName == "Pokaż utwory") this.CommandName = "Ukryj utwory";
                else this.CommandName = "Pokaż utwory";

                if (this.CommandVisibility == Visibility.Hidden)
                {
                    this.CommandVisibility = Visibility.Visible;
                    this.EditAlbumVisibility = Visibility.Hidden;
                }
                else
                {
                    this.CommandVisibility = Visibility.Hidden;
                    this.EditSongVisibility = Visibility.Hidden;
                }
            }
        }

        private void AddAlbumAction(object obj)
        {
            if(this.Cds != null)
            {
                Płyta cd = new Płyta();
                cd.ID = IDSetter();
                this.Cds.Add(cd);
            }
        }

        private void RemoveAlbumAction(object obj)
        {
            if(this.Cds != null)
            this.Cds.Remove(this.Cd);
        }

        private void EditAlbumAction(object obj)
        {
            if(this.Cds != null && this.EditSongVisibility == Visibility.Hidden)
            {
                if (this.EditAlbumVisibility == Visibility.Hidden) this.EditAlbumVisibility = Visibility.Visible;
                else this.EditAlbumVisibility = Visibility.Hidden;
            }
        }

        private void AddSongAction(object obj)
        {
            if(this.Cds != null && this.Cd != null)
            {
                Utwór song = new Utwór();
                this.Cd.utwory.songs.Add(song);
            }
        }

        private void RemoveSongAction(object obj)
        {
            if (this.Cds != null && this.Cd != null)
            {
                this.Cd.utwory.songs.Remove(this.Song);
            }
        }

        private void EditSongAction(object obj)
        {
            if (this.Cds != null && this.CommandVisibility == Visibility.Visible && this.EditAlbumVisibility == Visibility.Hidden)
            {
                if (this.EditSongVisibility == Visibility.Hidden) this.EditSongVisibility = Visibility.Visible;
                else this.EditSongVisibility = Visibility.Hidden;
            }
        }

        private bool CanExecuteAlwaysTrue(object obj)
        {
            return true;
        }

        private void OnPropertyChanged([CallerMemberName] string propertyName = null)
        {
            if(PropertyChanged != null)
            {
                PropertyChanged(this, new PropertyChangedEventArgs(propertyName));
            }
        }

        private String IDSetter()
        {
            String uniqueId = "a";
            bool isAvailable = true;
            int i = 1;
            while(isAvailable)
            {
                isAvailable = false;
                uniqueId = "a";
                uniqueId += i.ToString();
                foreach(Płyta cd in this.Cds)
                {
                    if (cd.ID == uniqueId) isAvailable = true;
                }
                i += 1;
            }
            return uniqueId;
        }

        private void GenreSetters()
        {
            foreach(Płyta cd in this.Cds)
            {
                String id = cd.gatunekPłyty.typ;
                switch(id)
                {
                    case "1": 
                        cd.gatunekPłyty.content = this.Genres[0].content;
                        break;
                    case "2":
                        cd.gatunekPłyty.content = this.Genres[1].content;
                        break;
                    case "3":
                        cd.gatunekPłyty.content = this.Genres[2].content;
                        break;
                    case "4":
                        cd.gatunekPłyty.content = this.Genres[3].content;
                        break;
                    default:
                        cd.gatunekPłyty.content = this.Genres[0].content;
                        break;
                }
            }
        }

        private void RefreshGenre()
        {
            foreach(Płyta cd in this.Cds)
            {
                String genre = cd.gatunekPłyty.content;
                switch(genre)
                {
                    case "Rock":
                        cd.gatunekPłyty.typ = this.Genres[0].typ;
                        break;
                    case "Metal":
                        cd.gatunekPłyty.typ = this.Genres[1].typ;
                        break;
                    case "Punk":
                        cd.gatunekPłyty.typ = this.Genres[2].typ;
                        break;
                    case "Pop":
                        cd.gatunekPłyty.typ = this.Genres[3].typ;
                        break;
                    default:
                        cd.gatunekPłyty.content = this.Genres[0].content;
                        break;
                }
            }
        }
    }
}
