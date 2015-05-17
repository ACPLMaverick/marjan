package  
{
	import away3d.core.base.Geometry;
	import away3d.materials.MaterialBase;
	import away3d.entities.Mesh;
	import away3d.textures.BitmapTexture;
	import flash.text.TextFormatAlign;
	import flash.text.TextFormat;
	import flash.text.TextField;
	import flash.display.BitmapData;
	import away3d.materials.TextureMaterial;
	import flash.text.TextFieldAutoSize;
	import flash.utils.Timer;
	import flash.events.TimerEvent;
	import away3d.tools.helpers.MeshHelper;
	import flash.geom.Vector3D;
	
	public class FieldDescription extends SceneObject
	{
		private var desc:String;
		private var textPlane:SceneObject;
		private var opened:Boolean = false;
		
		private var timerScale:Timer;
		
		public function FieldDescription(geometry:Geometry, material:MaterialBase, name:String = null, desc:String = "SomeDescription") 
		{
			super(geometry, material, name, false);
			
			this.desc = desc;
			this.timerScale = new Timer(33, 16);
			
			Initialize();
		}
		
		private function Initialize() : void
		{
			var tempFormat:TextFormat = new TextFormat();
			tempFormat.font = "TW Cen MT Condensed";
			tempFormat.color = 0xAAAAAAFF;
			tempFormat.align = TextFormatAlign.JUSTIFY;
			tempFormat.size = 80;
			tempFormat.leftMargin = 10;
			tempFormat.rightMargin = 10;
			var tempTF:TextField = new TextField();
			tempTF.defaultTextFormat = tempFormat;
			tempTF.width = 2048;
			tempTF.height = 1024;
			tempTF.wordWrap = true;
			tempTF.text = this.desc;
			var bm:BitmapData = new BitmapData(2048, 1024, true, 0);
			bm.draw(tempTF);
			var texture:BitmapTexture = new BitmapTexture(bm, true);
			
			var mat:TextureMaterial = new TextureMaterial(texture, false, false , false);
			mat.alphaPremultiplied = false;
			mat.alphaBlending = true;
			mat.alphaThreshold = 0.01;
			
//			var pvm:Number = 30;
//			MeshHelper.applyPosition(this, 0, -pvm, 0);
//			this.position = new Vector3D(0, 2.5*pvm, 0);
			
			textPlane = new SceneObject(this.geometry, mat);
			textPlane.scaleX = 0.775;
			textPlane.scaleY = 0.775;
			textPlane.scaleZ = 0.775;
			textPlane.z = textPlane.z - 10;
			
			this.addChild(textPlane);
			
			this.scaleY = 0;
		}
		
		public function get Opened() : Boolean
		{
			return opened;
		}
		
		public function set Opened(value:Boolean) : void
		{
			if(opened != value)
			{
				timerScale.reset();
				if(timerScale.hasEventListener(TimerEvent.TIMER)) timerScale.removeEventListener(TimerEvent.TIMER, TimerHandler);
				timerScale.addEventListener(TimerEvent.TIMER, TimerHandler, false, 0, true);
				timerScale.start();
				
				if(value)
				{
					// open up
					
				}
				else
				{
					// close up
				}
			}
			
			opened = value;
		}
		
		private function TimerHandler(e:TimerEvent) : void
		{
			if(opened)
			{
				this.scaleY += 0.0625;
			}
			else
			{
				this.scaleY -= 0.0625;
			}
		}
	}
	
}
