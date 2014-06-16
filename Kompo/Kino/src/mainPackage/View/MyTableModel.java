package mainPackage.View;

import javax.swing.table.AbstractTableModel;

/**
 * Klasa odpowiada za tworzenie tablic w interfejsie graficznym z okreslona zawartoscia i nazwami kolumn.
 */
public class MyTableModel extends AbstractTableModel{

	private String[] columnNames = {"Tytu³", "Godzina", "Dzieñ", "Wolne miejsca", "Cena"};
	private Object[][] content = {{"Ojciec Chrzestny", "17:00", "10-06-2014", "4", "14.00"}, 
							   	 {"Taksówkarz", "15:00", "12-06-2014", "5", "15.00"}
	};
	private String type;
	
	/**
	 * Tworzy nowy obiekt typu MyTableModel.
	 */
	public MyTableModel(){};
	
	/**
	 * Tworzy nowy obiekt typu MyTableModel z okreslonymi parametrami.
	 *
	 * @param columnNames tablica nazw kolumn tabeli.
	 * @param content macierz Object[][] danych zawartych w tabeli.
	 * @param type typ tabeli.
	 */
	public MyTableModel(String[] columnNames, Object[][] content, String type){
		this.columnNames = columnNames;
		setContent(content);
		this.type = type;
	}
	
	/* (non-Javadoc)
	 * @see javax.swing.table.TableModel#getColumnCount()
	 */
	public int getColumnCount() {
		return columnNames.length;
	}

	/* (non-Javadoc)
	 * @see javax.swing.table.TableModel#getRowCount()
	 */
	public int getRowCount() {
		return content.length;
	}
	
	/* (non-Javadoc)
	 * @see javax.swing.table.AbstractTableModel#getColumnName(int)
	 */
	public String getColumnName(int col){
		return columnNames[col];
	}

	/* (non-Javadoc)
	 * @see javax.swing.table.TableModel#getValueAt(int, int)
	 */
	public Object getValueAt(int row, int col) {
		return content[row][col];
	}
	
	/* (non-Javadoc)
	 * @see javax.swing.table.AbstractTableModel#getColumnClass(int)
	 */
	@SuppressWarnings("unchecked")
	public Class getColumnClass(int c){
		return getValueAt(0, c).getClass();
	}    
	
	/* (non-Javadoc)
	 * @see javax.swing.table.AbstractTableModel#isCellEditable(int, int)
	 */
	public boolean isCellEditable(int row, int col) {
        if (col < 6) {
            return false;
        } else {
            return true;
        }
    }
	
	/**
	 * Ustawia zawartosc tabeli.
	 *
	 * @param newContent macierz Object[][] zawierajaca nowe dane.
	 */
	public void setContent(Object[][] newContent) { this.content = newContent; }
}
