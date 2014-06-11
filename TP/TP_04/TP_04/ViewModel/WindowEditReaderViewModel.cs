using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;

namespace TP_04.ViewModel
{
    public class WindowEditReaderViewModel : Common.BindableBase
    {
        private Model.ModelController modelController;
        private ObservableCollection<Model.Reader> prop;
        private int selectedItem;

        public string prevName {get; set;}
        public string prevSurname { get; set; }
        public string prevPesel{ get; set; }

        private Model.Book _reader;
        public Model.Book reader
        {
            get { return _reader; }
            set
            {
                if(_reader != value)
                {
                    _reader = value;
                    OnPropertyChanged("book");
                }
            }
        }

        public Common.DelegateCommand ButtonOKCommand { get; set; }
        public Common.DelegateCommand ButtonCancelCommand { get; set; }

        public WindowEditReaderViewModel()
        {
            ButtonOKCommand = new Common.DelegateCommand(ButtonOKClicked);
            ButtonCancelCommand = new Common.DelegateCommand(ButtonCancelClicked);
            this.modelController = null;
            this.prop = null;
            this.selectedItem = -1;
            prevName = "none";
            prevSurname = "none";
            prevPesel = "0";
        }

        public WindowEditReaderViewModel(Model.ModelController controller, ObservableCollection<Model.Reader> prop, int selectedItem)
        {
            ButtonOKCommand = new Common.DelegateCommand(ButtonOKClicked);
            ButtonCancelCommand = new Common.DelegateCommand(ButtonCancelClicked);
            this.modelController = controller;
            this.prop = prop;
            this.selectedItem = selectedItem;
            prevName = modelController.GetOffice().getReaderCollection().ElementAt(selectedItem).name;
            prevSurname = modelController.GetOffice().getReaderCollection().ElementAt(selectedItem).secondName;
            prevPesel = Convert.ToString(modelController.GetOffice().getReaderCollection().ElementAt(selectedItem).pesel);
        }

        public void ButtonOKClicked(object sender)
        {
            List<object> sentList = (List<object>)sender;
            TextBox tbName = (TextBox)sentList[0];
            TextBox tbSurname = (TextBox)sentList[1];
            TextBox tbPesel = (TextBox)sentList[2];
            string name = tbName.Text;
            string surname = tbSurname.Text;
            string pesel = tbPesel.Text;
            if (name == "") name = "NO TITLE";
            if (surname == "") surname = "NO AUTHOR";
            if (pesel == "") pesel = "-1";
            editReader(name,surname,Convert.ToDouble(pesel));
            GetActiveWindow().Close();
        }

        public void ButtonCancelClicked(object sender)
        {
            GetActiveWindow().Close();
        }

        private void editReader(string name, string surname, double pesel)
        {
            //Model.Reader readerCol = modelController.GetOffice().getReaderCollection().ElementAt(selectedItem);
            Model.Reader readerProp = prop.ElementAt(selectedItem);
            //readerCol.name = name;
            //readerCol.secondName = surname;
            //readerCol.pesel = pesel;
            readerProp.name = name;
            readerProp.secondName = surname;
            readerProp.pesel = pesel;
        }
    }
}
