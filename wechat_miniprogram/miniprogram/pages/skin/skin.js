var api = require("../../utils/api.js");
var roiTools = require("../../utils/roi.js");

Page({
  data: {
    picked: null,   // {path, name, size}
    thickness: 2,
    submitting: false,
    roiEnabled: false,
    roi: null,
    roiStyle: "",
  },

  chooseImage: function () {
    var self = this;
    wx.chooseMedia({
      count: 1,
      mediaType: ["image"],
      sourceType: ["album", "camera"],
      success: function (res) {
        var f = res.tempFiles && res.tempFiles[0];
        if (!f) return;
        if (f.size > 8 * 1024 * 1024) {
          wx.showToast({ title: "图片需在 8MB 以内", icon: "none" });
          return;
        }
        self.setData({
          picked: { path: f.tempFilePath, name: "photo.jpg", size: f.size },
          roi: null, roiStyle: "",
        });
      },
      fail: function (err) { api.pickFailToast("选择照片", err); },
    });
  },

  onThickness: function (e) {
    this.setData({ thickness: Number(e.detail.value) });
  },

  clearImage: function () {
    this._roiRect = null;
    this.setData({ picked: null, roi: null, roiStyle: "" });
  },

  toggleRoi: function (e) {
    this.setData({ roiEnabled: !!e.detail.value, roi: null, roiStyle: "" });
    this._roiStart = null;
  },
  onPageScroll: function (e) {
    if (this._roiRect) this._roiRect.top -= e.scrollTop - (this._scrollTop || 0);
    this._scrollTop = e.scrollTop;
  },
  onResize: function () { if (this.data.picked) this.measureRoi(); },
  measureRoi: function () {
    var self = this;
    wx.createSelectorQuery().in(this).select(".roi-surface").boundingClientRect(function (rect) {
      self._roiRect = rect;
    }).exec();
  },
  roiStart: function (e) {
    if (!this.data.roiEnabled || !this._roiRect || this.data.submitting) return;
    this._roiStart = roiTools.point(e.touches[0], this._roiRect);
    this.setData({ roi: null, roiStyle: "" });
  },
  roiMove: function (e) {
    if (!this._roiStart || !e.touches.length) return;
    var roi = roiTools.box(this._roiStart, roiTools.point(e.touches[0], this._roiRect));
    this.setData({ roi: roi, roiStyle: roiTools.style(roi) });
  },
  roiEnd: function (e) {
    if (!this._roiStart) return;
    if (e.changedTouches && e.changedTouches.length) this.roiMove({ touches: e.changedTouches });
    this._roiStart = null;
    var roi = this.data.roi;
    if (!roi || (roi.x1-roi.x0)*this._roiRect.width < 12 || (roi.y1-roi.y0)*this._roiRect.height < 12) {
      this.setData({ roi: null, roiStyle: "" });
      wx.showToast({ title: "请拖动选择更大的区域", icon: "none" });
    }
  },
  roiCancel: function () { this._roiStart = null; this.setData({ roi: null, roiStyle: "" }); },

  submit: function () {
    var self = this;
    if (!this.data.picked || this.data.submitting) return;
    if (this.data.roiEnabled && (!this.data.roi || this._roiStart)) {
      wx.showToast({ title: "请先在图片上拖动框选 ROI", icon: "none" });
      return;
    }
    this.setData({ submitting: true });
    api.submitTaskFlow({
      type: "skin",
      files: [{ path: this.data.picked.path, name: this.data.picked.name }],
      params: Object.assign({ thickness: this.data.thickness }, this.data.roiEnabled ? { roi: this.data.roi } : {}),
      title: this.data.picked.name,
    }).then(function (id) {
      wx.redirectTo({ url: "/pages/task/task?id=" + id });
    }).catch(function (err) {
      self.setData({ submitting: false });
      wx.showToast({ title: err.message || "提交失败", icon: "none" });
    });
  },
});
