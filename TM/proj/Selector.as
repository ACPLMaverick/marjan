package  
{
	import flash.net.URLRequest;
	import away3d.core.base.Geometry;
	import away3d.materials.TextureMaterial;
	import flash.geom.Vector3D;
	import away3d.core.math.Vector3DUtils;
	import flash.ui.MouseCursorData;
	import away3d.textures.Texture2DBase;
	import away3d.textures.BitmapTexture;
	import flash.display.BitmapData;
	import away3d.materials.MaterialBase;
	import flash.xml.XMLNode;
	import flash.utils.Dictionary;
	
	public class Selector extends SceneObject
	{
		private const radius:Number = 400;
		private const lerpStep:Number = 0.1;
		
		private var xml:XML;
		private var planes:Array;
		private var currentlySelected:InfoPlane = null;
		private var currentlySelectedID:uint = 0;
		
		private var relativeRotationY:Number = 0.0;
		private var myRotationY:Number = 0.0;
		
		private var rotateAsMouse:Boolean = false;
		private var floatFreely:Boolean = false;
		private var rotationDir:Number = 0;
		private var lastY:Number = 0;
		private var lerp:Number = 0;
		private var angleToRotate:Number = 0;
		private var tempRotation:Number = 0;
		
		public var enabled:Boolean = true;
		
		public function Selector(name:String, xml:XML) 
		{
			super(null, null, name, false);
			this.xml = xml;
			Generate();
		}
		
		public override function Update() : void
		{
			super.Update();
			
			for each(var p:InfoPlane in planes)
			{
				p.Update();
			}
			
			if(enabled || lerp > 0)
			{
				if(rotateAsMouse)
				{
					RotateAsMouse();
				}
				else if(floatFreely)
				{
					FloatFreely();
				}
				else if(lerp > 0)
				{
					rotationY = angleToRotate + (tempRotation - angleToRotate) * lerp;
					lerp -= lerpStep;
				}
			}
		}

		private function Generate() : void
		{
			planes = ReadXML();
			if(planes.length == 0) return;
			
			var vector:Vector3D = new Vector3D(0.0, 0.0, -radius);
			var startAngle:Number = 0.0;
			var angleStep:Number = 360 / planes.length;
		
			
			for each(var plane:InfoPlane in planes)
			{
				this.addChild(plane);
				var newVector:Vector3D = Vector3DUtils.rotatePoint(vector.clone(), new Vector3D(0.0, startAngle, 0.0));
				plane.position = newVector;
				startAngle += angleStep;
			}
			
			this.currentlySelectedID = 0;
			this.currentlySelected = planes[currentlySelectedID] as InfoPlane;
		}
		
		private function ReadXML() : Array
		{
			var pList:XMLList = xml.asset;
			var pListLength:uint = pList.length();
			var array:Array = new Array();
			
			if(pListLength == 0) return array;
			
			var geoArray:Array = new Array();
			geoArray.push(System.getInstance().Models["InfoPlaneGm"] as Geometry);
			geoArray.push(System.getInstance().Models["InfoPlaneTitleGm"] as Geometry);
			geoArray.push(System.getInstance().Models["InfoPlaneDescGm"] as Geometry);
			geoArray.push(System.getInstance().Models["InfoPlaneViewerGm"] as Geometry);
			geoArray.push(System.getInstance().Models["InfoPlaneBtnShowDescGm"] as Geometry);
			geoArray.push(System.getInstance().Models["InfoPlaneBtnShowModelGm"] as Geometry);
			geoArray.push(System.getInstance().Models["InfoPlaneBtnLeftGm"] as Geometry);
			geoArray.push(System.getInstance().Models["InfoPlaneBtnRightGm"] as Geometry);
			
			var matArray:Array = new Array();
			matArray.push(System.getInstance().Materials["InfoPlaneMtBackground"] as MaterialBase);
			matArray.push(System.getInstance().Materials["InfoPlaneMtTitle"] as MaterialBase);
			matArray.push(System.getInstance().Materials["InfoPlaneMtDesc"] as MaterialBase);
			matArray.push(System.getInstance().Materials["InfoPlaneMtViewer"] as MaterialBase);
			matArray.push(System.getInstance().Materials["InfoPlaneMtBtnShowDesc"] as MaterialBase);
			matArray.push(System.getInstance().Materials["InfoPlaneMtBtnShowModel"] as MaterialBase);
			matArray.push(System.getInstance().Materials["InfoPlaneMtBtnLeft"] as MaterialBase);
			matArray.push(System.getInstance().Materials["InfoPlaneMtBtnRight"] as MaterialBase);
			
			var plane:InfoPlane;
			
			for(var i:uint = 0; i < pListLength; ++i)
			{
				var imgArray:Array = null;
				var vidArray:Array = null;
				var meshPath:String = null;
				var meshParts:Array = null;
				var meshScale:Number = 1;
				
				if("pics" in pList[i])
				{
					// get pics here
					imgArray = new Array();
					
					for each(var pic:XML in pList[i].pics.pic)
					{
						if(pic.hasOwnProperty("@path"))
						{
							var pictures:Dictionary = System.getInstance().Pictures;
							var path:String = pic.@path;
							var obj:Object = pictures[path];
							var mat:TextureMaterial = obj as TextureMaterial
							imgArray.push(mat);
						}
					}
				}
				if("videos" in pList[i])
				{
					vidArray = new Array();
					for each(var vid:XML in pList[i].videos.video)
					{
						if(vid.hasOwnProperty("@path") && vid.hasOwnProperty("@thumb"))
						{
							 vidArray.push(System.getInstance().GetMaterialFromVideoPath(vid.@path, vid.@thumb));  
						}
					}
				}
				if("model" in pList[i])
				{
					if((pList[i].model).hasOwnProperty("@path") &&
					   (pList[i].model).hasOwnProperty("@meshes") &&
					   (pList[i].model).hasOwnProperty("@scale"))
					{
						meshPath = pList[i].model.@path;
						var tempStr:String = pList[i].model.@meshes;
						meshParts = tempStr.split(";");
						meshScale = (Number(pList[i].model.@scale));
					}
				}

				plane = new InfoPlane(geoArray, matArray, pList[i].name, pList[i].description, 
									  imgArray, vidArray, meshPath, meshParts, meshScale, "InfoPlane" + i.toString());
				//System.getInstance().Objects["InfoPlane" + i.toString()] = plane;
				array.push(plane);
			}
			
			return array;
		}
		
		private function RotateAsMouse() : void
		{
			var mX:Number = -MouseController.getInstance().relativeMouseX;
			mX *= 2.5;
			this.rotationY += mX;
		}
		
		private function FloatFreely() : void
		{
			//trace("floatin");
			this.floatFreely = false;
		}
		
		public function SelectClosest() : void
		{
			//trace("sc");
			var camVec:Vector3D = System.getInstance().Graphics.camera.position.clone();
			camVec.y = 0.0;
			var angle:Number;
			var minAngle:Number = 360;
			var pl:InfoPlane = null;
			for each(var plane:InfoPlane in planes)
			{
				angle = 180/Math.PI * Vector3DUtils.getAngle(camVec, plane.position);
				if(angle < minAngle)
				{
					minAngle = angle;
					pl = plane;
				}
			}
			
			if(pl != null)
			{
				Select(pl);
			}
		}
		
		public function Select(plane:InfoPlane)
		{
			this.currentlySelected = plane;
			var id:uint = 0;
			for each(var plz:InfoPlane in planes)
			{
				if(plz == plane)
					break;
				++id;
			}
			this.currentlySelectedID = id;
			
			var camVec:Vector3D = System.getInstance().Graphics.camera.position.clone();
			camVec.y = 0.0;
			angleToRotate = 180/Math.PI * Vector3DUtils.getAngle(camVec, plane.position);
			
			tempRotation = myRotationY;

			if(plane.position.x > 0)
			{
				//tempRotation = myRotationY;
				angleToRotate += myRotationY;
			}
			else
			{
				angleToRotate = -angleToRotate;
				//tempRotation = myRotationY;
				angleToRotate += myRotationY;
			}
			
			//trace(angleToRotate, tempRotation, myRotationY, plane.position, camVec);
			
			StopRotatingAsMouse();
			this.lerp = 1;
		}
		
		public function SelectAsMouse(dir:int) : void
		{
			var mX:Number = MouseController.getInstance().relativeMouseX;
			var newID:uint = 0;
			if((mX > 0 && dir > 0) || (mX < 0 && dir < 0))
			{
				newID = (this.currentlySelectedID + 1) % (planes.length);
				if(newID > planes.length - 1)
					newID = planes.length - 1;
			}
			else if((mX < 0 && dir > 0) || (mX > 0 && dir < 0))
			{
				newID = (this.currentlySelectedID - 1) % (planes.length);
				if(newID > planes.length - 1)
					newID = planes.length - 1;
			}
			
			Select(planes[newID] as InfoPlane);
		}
		
		public function StartRotatingAsMouse(dir:Number) : void
		{
			this.rotateAsMouse = true;
			this.floatFreely = false;
			this.rotationDir = dir;
		}
		
		public function StopRotatingAsMouse() : void
		{
			this.rotateAsMouse = false;
			this.floatFreely = true;
		}
		
		public function StartAnimation() : void
		{
			tempRotation = myRotationY;
			this.rotationY = 25;
			this.angleToRotate = 335;
			this.lerp = 1;
		}
		
		public override function EnableInteractivity() : void
		{
			this.enabled = true;
			for each(var value:InfoPlane in planes)
			{
				value.EnableInteractivity();
			}
		}
		
		public override function DisableInteractivity() : void
		{
			this.enabled = false;
			for each(var value:InfoPlane in planes)
			{
				value.DisableInteractivity();
			}
		}
		
		public function GetThatShitFixed() : void
		{
			if((this.currentlySelected.fieldViewer as FieldViewer).isZoomed = true)
			{
				(this.currentlySelected.fieldViewer as FieldViewer).isZoomed = false;
			}
			
			(this.currentlySelected.fieldViewer as FieldViewer).PauseVideoIfPlaying();
		}
		
		public override function get rotationY() : Number
		{
			return myRotationY;
		}
		
		public override function set rotationY(val:Number) : void
		{
			this.relativeRotationY = val - myRotationY;
			this.myRotationY = val;
			
			if(this.numChildren == 0) return;
			
			var so:SceneObject;
			for(var i:uint = 0; i < numChildren; ++i)
			{
				so = getChildAt(i) as SceneObject;
				so.position = Vector3DUtils.rotatePoint(so.position, new Vector3D(0.0, relativeRotationY, 0.0));
			}
		}
		
		public function get CurrentlySelected() : InfoPlane
		{
			return this.currentlySelected;
		}
	}
	
}
